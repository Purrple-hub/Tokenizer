"""CTypes wrapper module for Tokenizer project.

Provides interface for C/C++ library integration, low-level operations,
and cross-language compatibility for tokenization and memory management."""
import ctypes
import os
from typing import Optional, List, Dict, Any, Union, Callable

class CTypesWrapper:
    """Wrapper for ctypes integration with C/C++ libraries.
    
    Provides interface for low-level tokenization operations,
    memory management, and cross-language compatibility."""
    
    def __init__(self, library_path: Optional[str] = None):
        self.lib = None
        self._loaded = False
        if library_path and os.path.exists(library_path):
            self.load_library(library_path)
    
    def load_library(self, library_path: str) -> bool:
        """Load a shared library (.dll, .so, .dylib).
        
        Args:
            library_path: Path to the shared library
            
        Returns:
            True if library loaded successfully
        """
        try:
            self.lib = ctypes.CDLL(library_path)
            self._loaded = True
            self._log(f"Library loaded: {library_path}")
            return True
        except OSError as e:
            print(f"Failed to load library {library_path}: {e}")
            self.lib = None
            self._loaded = False
            return False
    
    def set_tokenizer_callback(self, 
                             tokenizer_ptr: int, 
                             text_ptr: int, 
                             text_len: int,
                             output_buf_size: int = 4096) -> int:
        """Set up callback for tokenizer operation via C library.
        
        Args:
            tokenizer_ptr: Pointer to tokenizer instance
            text_ptr: Pointer to input text data
            text_len: Length of text data
            output_buf_size: Size of output buffer
            
        Returns:
            Result code from C function
        """
        if not self._loaded or self.lib is None:
            return -1
        
        if hasattr(self.lib, 'tokenize_callback'):
            self.lib.tokenize_callback.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int
            ]
            self.lib.tokenize_callback.restype = ctypes.c_int
            
            output_buffer = ctypes.create_string_buffer(output_buf_size)
            result = self.lib.tokenize_callback(
                ctypes.c_void_p(tokenizer_ptr),
                ctypes.c_char_p(text_ptr),
                ctypes.c_int(text_len),
                output_buffer,
                ctypes.c_int(output_buf_size)
            )
            return result
        return -1
    
    def memmove(self, dst: int, src: bytes, size: int) -> bool:
        """Copy memory block from source to destination address.
        
        Args:
            dst: Destination address
            src: Source bytes
            size: Number of bytes to copy
            
        Returns:
            True if successful
        """
        if not self._loaded or self.lib is None:
            try:
                ctypes.memmove(dst, src, size)
                return True
            except Exception:
                return False
        
        if hasattr(self.lib, 'memmove'):
            self.lib.memmove.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            self.lib.memmove.restype = None
            ctypes.memmove(dst, src, size)
            return True
        return False
    
    def create_buffer(self, size: int) -> ctypes.Array:
        """Create a byte buffer of specified size.
        
        Args:
            size: Buffer size in bytes
            
        Returns:
            ctypes byte array buffer
        """
        return ctypes.create_string_buffer(size)
    
    def set_string_attribute(self, obj: int, attr_name: str, value: str) -> bool:
        """Set a string attribute on a ctypes object.
        
        Args:
            obj: Object pointer
            attr_name: Attribute name
            value: String value
            
        Returns:
            True if successful
        """
        if not self._loaded or self.lib is None:
            try:
                return True
            except Exception:
                return False
        
        if hasattr(self.lib, 'set_string_attribute'):
            self.lib.set_string_attribute.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
            self.lib.set_string_attribute.restype = ctypes.c_bool
            return self.lib.set_string_attribute(
                ctypes.c_void_p(obj),
                attr_name.encode('utf-8'),
                ctypes.c_int(len(value))
            )
        return False
    
    def _log(self, message: str, level: str = "info"):
        """Internal logging."""
        print(f"[CTypesWrapper] {level.upper()}: {message}")

_ctypes_wrapper = CTypesWrapper()

def get_ctypes_wrapper() -> CTypesWrapper:
    """Get the global ctypes wrapper instance."""
    return _ctypes_wrapper

def load_c_library(library_path: str) -> Optional[CTypesWrapper]:
    """Load a C shared library and return the wrapper.
    
    Args:
        library_path: Path to .dll or .so file
        
    Returns:
        CTypesWrapper instance or None if failed
    """
    wrapper = CTypesWrapper(library_path)
    if wrapper._loaded:
        return wrapper
    return None
