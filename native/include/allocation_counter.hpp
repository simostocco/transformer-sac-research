#pragma once
#include <atomic>
#include <cstdlib>
#include <new>
std::atomic<size_t> allocations=0,c_allocations=0;
#if defined(__linux__) && !defined(__SANITIZE_ADDRESS__)
extern "C" void* __libc_malloc(size_t);
extern "C" void* __libc_calloc(size_t,size_t);
extern "C" void* __libc_realloc(void*,size_t);
extern "C" void* malloc(size_t n){c_allocations.fetch_add(1,std::memory_order_relaxed);return __libc_malloc(n);}
extern "C" void* calloc(size_t n,size_t size){c_allocations.fetch_add(1,std::memory_order_relaxed);return __libc_calloc(n,size);}
extern "C" void* realloc(void* p,size_t size){c_allocations.fetch_add(1,std::memory_order_relaxed);return __libc_realloc(p,size);}
#endif
void* operator new(std::size_t n){allocations.fetch_add(1,std::memory_order_relaxed);if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
