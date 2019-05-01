#pragma once

#include <iostream>
#include <atomic>

#include <emmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>

constexpr uint64_t cacheLine = 64;
constexpr uint64_t cacheLineMask = 63;
class RingBuff
{
  public:
    using RingBuff_t = std::unique_ptr<char[]>;

  private:
    const int32_t ringBuffSize_{0};
    const int32_t ringBuffMask_{0};
    const int32_t ringBuffOverflow_{1024};

    RingBuff_t ringBuff0_;
    char* const ringBuff_;

    std::atomic<int32_t> atomicHead_{0};
    int32_t head_{0};
    int32_t lastFlushedHead_{0};
    //    alignas(64) char x;
    std::atomic<int32_t> atomicTail_{0};
    int32_t tail_{0};
    int32_t lastFlushedTail_{0};

  public:

    RingBuff() : RingBuff(1024) {}
    RingBuff(uint32_t sz) 
      : ringBuffSize_(sz)
        , ringBuffMask_(ringBuffSize_-1)
        , ringBuff0_(new 
            char[ringBuffSize_+ringBuffOverflow_])
        , ringBuff_{(char*)(((intptr_t)
              (ringBuff0_.get()) + cacheLineMask) & 
            ~(cacheLineMask))}
    {
      for ( int i = 0; 
          i < ringBuffSize_+ringBuffOverflow_; 
          ++i)
        ringBuff_[i] = '5';

      memset( ringBuff_, 
              0, 
              ringBuffSize_+ringBuffOverflow_);

      // eject log memory from cache
      for ( int i = 0; 
            i <ringBuffSize_+ringBuffOverflow_; 
            i+= cacheLine)
        _mm_clflush(ringBuff_+i); 

      // load first 100 cache lines into memory
      for (int i = 0; i < 100; ++i)
        _mm_prefetch( ringBuff_ + 
            (i*cacheLine), 
            _MM_HINT_T0);
    }

    ~RingBuff() 
    {
    }

    // TODO:FIXME do proper bounds checking later
    int32_t getHead( int32_t diff = 0 ) 
    { return (head_+diff) & ringBuffMask_; }
    int32_t getTail( int32_t diff = 0 ) 
    { return (tail_+diff) & ringBuffMask_; }

    char* pickProduce (int32_t sz = 0) 
    {
      auto ft = atomicTail_.load(
          std::memory_order_acquire);

      return (head_ - ft > ringBuffSize_ - 
          (128+sz)) ? nullptr : 
            ringBuff_ + getHead(); 
    }

    char* pickConsume (int32_t sz = 0) 
    {
      auto fh = atomicHead_.load(
          std::memory_order_acquire); 
      return fh - (tail_+sz) < 1 ? nullptr :
        ringBuff_ + getTail(); 
    }

    void produce ( uint32_t sz ) { head_ += sz; }
    void consume ( uint32_t sz ) { tail_ += sz; }


    uint32_t clfuCount{0};
    void cleanUp(int32_t& last, int32_t offset)
    {
      auto lDiff = last - (last & cacheLineMask);
      auto cDiff = offset - 
        (offset & cacheLineMask);

      while (cDiff > lDiff)
      {
        _mm_clflushopt(ringBuff_ + 
            (lDiff & ringBuffMask_));

        lDiff += cacheLine;
        last = lDiff;

        ++clfuCount;
      }
    }

    void cleanUpConsume()
    {
      cleanUp(lastFlushedTail_, tail_);

      atomicTail_.store(tail_, 
          std::memory_order_release);
    }

    void cleanUpProduce()
    {
      cleanUp(lastFlushedHead_, head_);

      // signifigant improvement to fat tails
      _mm_prefetch(ringBuff_ + 
          getHead(cacheLine*12), _MM_HINT_T0); 
      // move before prefetch?
      atomicHead_.store(head_, 
          std::memory_order_release);
    }

    char* get() { return ringBuff_; }
};

