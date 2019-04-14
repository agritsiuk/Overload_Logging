#pragma once

#include <iostream>
#include <atomic>

#include <emmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>

///////////////////////////////////////////////////////////////////////////////

class SimpleRingBuff
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
    alignas(64) char x;
    std::atomic<int32_t> atomicTail_{0};
    int32_t tail_{0};
    int32_t lastFlushedTail_{0};

public:
    void dbgPrint ()
    {
        std::cout << "Raw Head = " << head_ << std::endl;
        std::cout << "Raw Tail = " << tail_ << std::endl;
        std::cout << "Head = " << getHead() << std::endl;
        std::cout << "Tail = " << getTail() << std::endl;
        std::cout << "Diff = " << getHead() - getTail() << std::endl;
        std::cout << "raw flushedHead = " << (atomicHead_.load()) << std::endl;
        std::cout << "raw flushedTail = " << (atomicTail_.load()) << std::endl;
        std::cout << "flushedHead = " << (atomicHead_.load() & ringBuffMask_) << std::endl;
        std::cout << "flushedTail = " << (atomicTail_.load() & ringBuffMask_) << std::endl;
        std::cout << "flushed Diff = " << atomicHead_.load() - atomicTail_.load() << std::endl;
        std::cout << "Produce hit = " << fp << ", miss = " << fp_miss << std::endl;
        std::cout << "Consume hit = " << fc << ", miss = " << fc_miss << std::endl;
    }

    SimpleRingBuff() : SimpleRingBuff(1024) {}
    SimpleRingBuff(uint32_t sz) 
        : ringBuffSize_(sz)
        , ringBuffMask_(ringBuffSize_-1)
        , ringBuff0_(new char[ringBuffSize_+ringBuffOverflow_])
        , ringBuff_{(char*)(((intptr_t)(ringBuff0_.get()) + 63) & ~(63ull))}
    {
        for (int i = 0; i < ringBuffSize_+ringBuffOverflow_; ++i)
            ringBuff_[i] = '5';

        memset(ringBuff_, 0, ringBuffSize_+ringBuffOverflow_);

        // eject log memory from cache
        //for (int i = 0; i <ringBuffSize_+ringBuffOverflow_; i+= 64)
        //    _mm_clflush(ringBuff_+i);
    }

    // TODO:FIXME do proper bounds checking later
    int32_t getHead( int32_t diff = 0 ) { return (head_+diff) & ringBuffMask_; }
    int32_t getTail( int32_t diff = 0 ) { return (tail_+diff) & ringBuffMask_; }

    int32_t getFreeSpace () 
    {
        return ringBuffSize_ - (atomicHead_.load(std::memory_order_relaxed) - atomicTail_.load(std::memory_order_relaxed)); 
    }

    char* pickProduce (int32_t sz = 0) 
    {
        auto ft = atomicTail_.load(std::memory_order_acquire); 
        return (head_ - ft > ringBuffSize_ - (128+sz)) ? nullptr : ringBuff_ + getHead(); 
    }

    char* pickConsume (int32_t sz = 0) 
    {
        auto fh = atomicHead_.load(std::memory_order_acquire); 
        return fh - (tail_+sz) < 1 ? nullptr : ringBuff_ + getTail(); 
    }

    void produce ( uint32_t sz ) { head_ += sz; }
    void consume ( uint32_t sz ) { tail_ += sz; }


    void cleanUp(int32_t& last, int32_t offset)
    {
#if defined(FP_AUX) || defined(LOG_NTS)
        return;
#endif

#if defined(LOG_CLFLUSHOPT1) || defined(LOG_CLFLUSHOPT2) || defined(LOG_CLFLUSHOPT3)
        auto lDiff = last - (last % 64);
        auto cDiff = offset - (offset % 64);

        while (cDiff > lDiff)
        {
            //_mm_clflushopt(ringBuff_ + (lDiff & ringBuffMask_));

            //_mm_clflush(data+lDiff);
            
            lDiff += 64;
            last = lDiff;
        }
#endif
    }

    void cleanUpConsume()
    {
        cleanUp(lastFlushedTail_, tail_);

        // consumption is typically on the slow path, don't need this optimization
        // is std::memory_order_release sufficent?
        atomicTail_.store(tail_, std::memory_order_release);
    }

    void cleanUpProduce()
    {
        cleanUp(lastFlushedHead_, head_);

        // _MM_HINT_T1 we can avoid polluting L3 alltogether wit T1? possibly, load into L2 and L1 
        // flush soon after, never touches L3?
        _mm_prefetch(ringBuff_ + getHead(64*4), _MM_HINT_T0); // about 6-7 cpu cycle improvement (@ 10%)
        _mm_prefetch(ringBuff_ + getHead(64*8), _MM_HINT_T0); // about 6-7 cpu cycle improvement (@ 10%)
        _mm_prefetch(ringBuff_ + getHead(64*12), _MM_HINT_T0); // about 6-7 cpu cycle improvement (@ 10%)
        // is std::memory_order_release sufficent?
        atomicHead_.store(head_, std::memory_order_release);
    }

    // flush methods are for FP_AUX
    void flush(int32_t& last, int32_t offset)
    {
        auto lDiff = last - (last % 64);
        auto cDiff = offset - (offset % 64);

        while (cDiff > lDiff)
        {
            _mm_clflush(ringBuff_ + (lDiff & ringBuffMask_));
            
            lDiff += 64;
            last = lDiff;
        }
    }

    uint32_t fc{0};
    uint32_t fc_miss{0};
    int32_t flushConsume( int32_t last )
    {
        auto ft = atomicTail_.load(std::memory_order_acquire);
        if (ft - last >= 64)
            ++fc;
        else
            ++fc_miss;

        if (last >= ft)
            return ft;

        flush(last, ft);
        // is std::memory_order_release sufficent?
        atomicTail_.store(last, std::memory_order_release);

        return ft;
    }

    uint32_t fp{0};
    uint32_t fp_miss{0};
    int32_t flushProduce( int32_t last )
    {
        auto fh = atomicHead_.load(std::memory_order_acquire);
        if (fh - last >= 64)
            ++fp;
        else
            ++fp_miss;

        flush(last, fh);
        // _MM_HINT_T1 we can avoid polluting L3 alltogether wit T1? possibly, load into L2 and L1 
        // flush soon after, never touches L3?
        _mm_prefetch(ringBuff_ + getHead(64*4), _MM_HINT_T0); // about 6-7 cpu cycle improvement (@ 10%)
        // is std::memory_order_release sufficent?
        atomicHead_.store(head_, std::memory_order_release);

        return fh;
    }



    char* get() { return ringBuff_; }

    void reset()
    {
        for (int i = 0; i < ringBuffSize_; ++i)
            ringBuff_[i] = '5';

        memset(ringBuff_, 0, ringBuffSize_);

        // eject log memory from cache
        for (int i = 0; i <ringBuffSize_; ++i)
            _mm_clflush(ringBuff_+i);

        head_ = 0;
        tail_ = 0;
        lastFlushedHead_ = 0;
        lastFlushedTail_ = 0;
    }
};

