import re
import hashlib
from typing import List, Optional, Any

class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, pages: list, document_id: str, 
                       document_type: str = '', 
                       equipment_ids: Optional[List[str]] = None) -> list:
        '''Chunk document pages into overlapping text chunks.
        Each chunk preserves: document_id, page_number, equipment_ids.
        Equipment IDs are detected per-chunk AND inherited from document-level.
        '''
        from .schemas import Chunk
        doc_equipment_ids = equipment_ids or []
        chunks = []
        chunk_idx = 0
        
        for page in pages:
            text = getattr(page, 'text', '') if hasattr(page, 'text') else page.get('text', '')
            page_number = getattr(page, 'page_number', 1) if hasattr(page, 'page_number') else page.get('page_number', 1)
            
            page_chunks = self.chunk_text(
                text=text,
                document_id=document_id,
                page_number=page_number,
                document_type=document_type,
                equipment_ids=doc_equipment_ids
            )
            
            for c_data in page_chunks:
                c_data['chunk_index'] = chunk_idx
                c_data['chunk_id'] = self._generate_chunk_id(document_id, chunk_idx)
                chunks.append(Chunk(**c_data))
                chunk_idx += 1
                
        return chunks
    
    def chunk_text(self, text: str, document_id: str, page_number: int = 1,
                   document_type: str = '', equipment_ids: Optional[List[str]] = None) -> list:
        '''Chunk a single text string.'''
        doc_equipment_ids = equipment_ids or []
        text_chunks = self._split_text(text)
        
        results = []
        for t in text_chunks:
            local_eq_ids = self._detect_equipment_ids(t)
            all_eq_ids = sorted(list(set(doc_equipment_ids + local_eq_ids)))
            
            results.append({
                'chunk_id': '', # Filled later
                'text': t,
                'document_id': document_id,
                'page_number': page_number,
                'chunk_index': 0, # Filled later
                'equipment_ids': all_eq_ids,
                'document_type': document_type,
                'metadata': {}
            })
        return results
    
    def _split_text(self, text: str) -> list[str]:
        '''Split text into chunks with overlap.
        Strategy: split by paragraphs first, then by sentences if paragraph too long.
        '''
        words = text.split()
        chunks = []
        if not words:
            return chunks
        
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_len = len(word) + 1 # +1 for space
            if current_length + word_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Backtrack for overlap
                overlap_words = []
                overlap_length = 0
                for w in reversed(current_chunk):
                    if overlap_length + len(w) + 1 > self.chunk_overlap:
                        break
                    overlap_words.insert(0, w)
                    overlap_length += len(w) + 1
                current_chunk = overlap_words
                current_length = overlap_length
            
            current_chunk.append(word)
            current_length += word_len
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def _detect_equipment_ids(self, text: str) -> list[str]:
        '''Find equipment IDs using regex: patterns like V-101, P-101, HX-101, FV-103.
        Returns deduplicated sorted list.
        '''
        pattern = r'\b[A-Z]{1,3}-\d{2,4}\b'
        matches = re.findall(pattern, text)
        return sorted(list(set(matches)))
    
    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        '''Generate deterministic chunk ID.'''
        s = f"{document_id}_{chunk_index}"
        return hashlib.md5(s.encode('utf-8')).hexdigest()
