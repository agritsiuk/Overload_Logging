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

    std::atomic<int32_t> flushedHead_{0};
    int32_t head_{0};
    int32_t lastHead_{0};
    alignas(64) char x;
    std::atomic<int32_t> flushedTail_{0};
    int32_t tail_{0};
    int32_t lastTail_{0};

public:
    void dbgPrint ()
    {
        std::cout << "Raw Head = " << head_ << std::endl;
        std::cout << "Raw Tail = " << tail_ << std::endl;
        std::cout << "Head = " << getHead() << std::endl;
        std::cout << "Tail = " << getTail() << std::endl;
        std::cout << "Diff = " << getHead() - getTail() << std::endl;
        std::cout << "flushedHead = " << (flushedHead_.load() & ringBuffMask_) << std::endl;
        std::cout << "flushedTail = " << (flushedTail_.load() & ringBuffMask_) << std::endl;
        std::cout << "flushed Diff = " << flushedHead_.load() - flushedTail_.load() << std::endl;
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
        for (int i = 0; i <ringBuffSize_+ringBuffOverflow_; i+= 64)
            _mm_clflushopt(ringBuff_+i);
    }

    // do proper bounds checking later

    int32_t getHead( int32_t diff = 0 ) { return (head_+diff) & ringBuffMask_; }
    int32_t getTail( int32_t diff = 0 ) { return (tail_+diff) & ringBuffMask_; }

    char* pickProduce (int32_t sz = 0) 
    { auto ft = flushedTail_.load(std::memory_order_acquire); return (head_ - ft > ringBuffSize_ - 128) ? nullptr : ringBuff_ + getHead(); }

    char* pickConsume (int32_t sz = 0) 
    { auto fh = flushedHead_.load(std::memory_order_acquire); return fh - tail_ < 1 ? nullptr : ringBuff_ + getTail(); }
    //
    //char* pickProduce () { return head_ - tail_ > ringBuffSize_ - 128 ? nullptr : ringBuff_ + getHead(); }
    //char* pickConsume () { return head_ - tail_ < 1 ? nullptr : ringBuff_ + getTail(); }

    //char* pickConsume () { auto r = head_ - tail_; std::cout << " '" << r << "'"; return r < 1 ? nullptr : ringBuff_ + getTail(); }

    void produce ( uint32_t sz ) { head_ += sz; }
    void consume ( uint32_t sz ) { tail_ += sz; }


    void cleanUp(int32_t& last, int32_t offset)
    {
#ifdef FP_AUX
        return;
#endif
        auto lDiff = last - (last % 64);
        auto cDiff = offset - (offset % 64);

        while (cDiff > lDiff)
        {
            //std::cout << std::endl;
            //std::cout << "Flushing " << lDiff << " / " << lDiff / 64 << ", current = " 
            //         << getTail() << ", diff = " << getTail()-lDiff << std::endl;
            _mm_clflushopt(ringBuff_ + (lDiff & ringBuffMask_)); // 60 - 62 cpu cycles on L29

            //_mm_clflush(data+lDiff); // 280 cpu cycles on L29
            
            lDiff += 64;
            last = lDiff;
        }
        //std::cout << "lastOffset = " << lastHead_ << ", current = " 
        //          << getHead() << ", diff = " << getHead() - lastHead_ 
        //          << " cDiff = " << cDiff << ", lDiff = " << lDiff << std::endl;
    }

    void cleanUpConsume()
    {
        cleanUp(lastTail_, tail_);
        // consumption is typically on the slow path, don't need this optimization
        // If T1 is L2 and higher then L3 is untouched!  This is not a speed optimizatoin
        // but a zero touch L3?
        _mm_prefetch(ringBuff_ + getTail()  +(64*4), _MM_HINT_T1); // about 6-7 cpu cycle improvement (@ 10%)
        // is memory_order_release sufficent?
        flushedTail_.store(tail_, std::memory_order_release);
    }

    void cleanUpProduce()
    {
#if defined(LOG_CLFLUSHOPT1) || defined(LOG_CLFLUSHOPT2) || defined(LOG_CLFLUSHOPT3)
        cleanUp(lastHead_, head_);
#endif
        // _MM_HINT_T1 we can avoid polluting L3 alltogether wit T1?
        _mm_prefetch(ringBuff_ + getHead(64*4), _MM_HINT_T1); // about 6-7 cpu cycle improvement (@ 10%)
        // is memory_order_release sufficent?
        flushedHead_.store(head_, std::memory_order_release);
    }

    void flush(int32_t& last, int32_t offset)
    {
        auto lDiff = last - (last % 64);
        auto cDiff = offset - (offset % 64);

        //std::cerr   << "lastOffset = " << last << ", offset = " << offset 
        //            << " lDiff = " << lDiff << ", cDiff = " << cDiff
        //            << std::endl;

        while (cDiff > lDiff)
        {
            //std::cerr << std::endl;
            //std::cerr << "Flushing " << lDiff << " / " << lDiff / 64 << ", cDiff = " << cDiff
            //    << ", current = " << getTail() << ", diff = " << getTail()-lDiff << std::endl;
            _mm_clflush(ringBuff_ + (lDiff & ringBuffMask_));

            
            lDiff += 64;
            last = lDiff;
        }
    }

    uint32_t fc{0};
    uint32_t fc_miss{0};
    int32_t flushConsume( int32_t last )
    {
        auto ft = flushedTail_.load(std::memory_order_acquire);
        if (ft - last >= 64)
            ++fc;
        else
            ++fc_miss;
        /*
        ++fc;
        if (ft > last)
            std::cerr << "flushConsume, last = " << last << ", tail = " << ft << ", total " << ft - last << ", CLs " << (ft-last)/64 << ", miss = " << fc_miss << std::endl;
        else
            ++fc_miss;
        // */

        if (last >= ft)
            return ft;

        flush(last, ft);
        // consumption is typically on the slow path, don't need this optimization
        // If T1 is L2 and higher then L3 is untouched!  This is not a speed optimizatoin
        // but a zero touch L3?
        _mm_prefetch(ringBuff_ + getTail()  +(64*4), _MM_HINT_T0); // about 6-7 cpu cycle improvement (@ 10%)
        // is memory_order_release sufficent?
        flushedTail_.store(last, std::memory_order_release);

        return ft;
    }

    uint32_t fp{0};
    uint32_t fp_miss{0};
    int32_t flushProduce( int32_t last )
    {
        auto fh = flushedHead_.load(std::memory_order_acquire);
        if (fh - last >= 64)
            ++fp;
        else
            ++fp_miss;
        /*
        ++fp;
        if (fh > last)
            std::cerr << "flushProduce, last = " << last << ", tail = " << fh << ", total " << fh - last << ", CLs " << (fh-last)/64 << ", miss = " << fp_miss << std::endl;
        else
            ++fp_miss;

        if (last >= fh)
            return fh;
        // */

        flush(last, fh);
        // _MM_HINT_T1 we can avoid polluting L3 alltogether wit T1?
        _mm_prefetch(ringBuff_ + getHead(64*4), _MM_HINT_T0); // about 6-7 cpu cycle improvement (@ 10%)
        // is memory_order_release sufficent?
        flushedHead_.store(head_, std::memory_order_release);

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
            _mm_clflushopt(ringBuff_+i);

        head_ = 0;
        tail_ = 0;
        lastHead_ = 0;
        lastTail_ = 0;
    }
};

