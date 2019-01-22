#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <functional>
#include <tuple>
#include <type_traits>

#include <unistd.h>
#include <signal.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>

// uncomment to compile for use of CLFLUSHOPT rather than NTS
//#define LOG_CLFLUSHOPT
//*
uint64_t getcc_b ( void )
{
	unsigned cycles_low, cycles_high;

	asm volatile (	
					"CPUID\n\t"
					"RDTSCP\n\t"
					"mov %%edx, %0\n\t"
					"mov %%eax, %1\n\t" : "=r" (cycles_high), "=r" (cycles_low)::
					"%rax", "%rbx", "%rcx", "%rdx");

	return ((uint64_t)cycles_high << 32 | cycles_low);
}

uint64_t getcc_e ( void )
{
	unsigned cycles_low, cycles_high;

	asm volatile (
					"RDTSCP\n\t"
					"mov %%edx, %0\n\t"
					"mov %%eax, %1\n\t CPUID\n\t" : "=r" (cycles_high), "=r" (cycles_low)::
					"%rax", "%rbx", "%rcx", "%rdx");

	return ((uint64_t)cycles_high << 32 | cycles_low);
}

void wait ( uint32_t cycles )
{
    auto cur = __rdtsc();
    while(__rdtsc() < cur + cycles){}
}

void setAffinity(	
		  std::thread& t 
		, uint32_t cpuid )
{
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(cpuid, &cpuset);

    int rc = pthread_setaffinity_np(
			t.native_handle()
			, sizeof(cpu_set_t)
			, &cpuset);

	std::cerr	<< "affinity " 
				<< cpuid 
				<< std::endl;

	if (rc != 0) 
	{
		std::cerr << "Error calling "
					 "pthread_setaffinity_np: "
				  << rc 
				  << "\n";
		exit (0);
	}
}

///////////////////////////////////////////////////////////////////////////////

class SimpleRingBuff
{
public:
    using RingBuff_t = std::unique_ptr<char[]>;
private:
    const int32_t ringBuffSize_{0};
    const int32_t ringBuffMask_{0};
    const int32_t ringBuffOverflow_{1024};

    RingBuff_t ringBuff_;

    alignas(64) std::atomic<int32_t> finalHead_{0};
    alignas(64) char x;
    alignas(64) std::atomic<int32_t> finalTail_{0};

    //alignas(64) std::atomic<int32_t> head_{0};
    //alignas(64) char x;
    //alignas(64) std::atomic<int32_t> tail_{0};

    struct
    {
        alignas(64) int32_t head_{0};
        alignas(64) char x2;
        alignas(64) int32_t tail_{0};

        alignas(64) int32_t lastHead_{0};
        alignas(64) char x3;
        alignas(64) int32_t lastTail_{0};
    };

public:
    SimpleRingBuff() : SimpleRingBuff(1024) {}
    SimpleRingBuff(uint32_t sz) 
        : ringBuffSize_(sz)
        , ringBuffMask_(ringBuffSize_-1)
        , ringBuff_(new char[ringBuffSize_+ringBuffOverflow_])
    {
        for (int i = 0; i < ringBuffSize_+ringBuffOverflow_; ++i)
            ringBuff_[i] = '5';

        memset(ringBuff_.get(), 0, ringBuffSize_+ringBuffOverflow_);

        // eject log memory from cache
        for (int i = 0; i <ringBuffSize_+ringBuffOverflow_; i+= 64)
            _mm_clflushopt(ringBuff_.get()+i);
    }

    // do proper bounds checking later

    int32_t getHead( int32_t diff = 0 ) { return (head_+diff) & ringBuffMask_; }
    int32_t getTail( int32_t diff = 0 ) { return (tail_+diff) & ringBuffMask_; }

    char* pickProduce () { auto ft = finalTail_.load(std::memory_order_acquire); return (head_ - ft > ringBuffSize_ - 128) ? nullptr : ringBuff_.get() + getHead(); }
    char* pickConsume () { auto fh = finalHead_.load(std::memory_order_acquire); return fh - tail_ < 1 ? nullptr : ringBuff_.get() + getTail(); }
    //
    //char* pickProduce () { return head_ - tail_ > ringBuffSize_ - 128 ? nullptr : ringBuff_.get() + getHead(); }
    //char* pickConsume () { return head_ - tail_ < 1 ? nullptr : ringBuff_.get() + getTail(); }

    //char* pickConsume () { auto r = head_ - tail_; std::cout << " '" << r << "'"; return r < 1 ? nullptr : ringBuff_.get() + getTail(); }

    void produce ( uint32_t sz ) { head_ += sz; }
    void consume ( uint32_t sz ) { tail_ += sz; }


    void cleanUp(int32_t& last, int32_t offset)
    {
        auto lDiff = last - (last % 64);
        auto cDiff = offset - (offset % 64);

        while (cDiff > lDiff)
        {
            //std::cout << std::endl;
            //std::cout << "Flushing " << lDiff << " / " << lDiff / 64 << ", current = " 
            //         << getTail() << ", diff = " << getTail()-lDiff << std::endl;
            _mm_clflushopt(ringBuff_.get() + (lDiff & ringBuffMask_)); // 60 - 62 cpu cycles on L29

            //_mm_clflush(data+lDiff); // 280 cpu cycles on L29
            
            lDiff += 64;
            last = lDiff;
        }
        //std::cout << "lastOffset = " << lastHead_ << ", current = " 
        //          << getHead() << ", diff = " << getHead() - lastHead_ 
        //          << " cDiff = " << cDiff << ", lDiff = " << lDiff << std::endl;
    }

    void cleanUpProduce()
    {
#ifdef LOG_CLFLUSHOPT1
        cleanUp(lastHead_, head_);
#endif
// HAX!1!!!
#ifdef LOG_CLFLUSHOPT2
        cleanUp(lastHead_, head_);
#endif
        _mm_prefetch(ringBuff_.get() + getHead(64*10), _MM_HINT_T0); // about 6-7 cpu cycle improvement (@ 10%)
        // is memory_order_release sufficent?
        finalHead_.store(head_, std::memory_order_release);
    }

    void cleanUpConsume()
    {
        cleanUp(lastTail_, tail_);
        // consumption is typically on the slow path, don't need this optimization
        //_mm_prefetch(ringBuff_.get() + getTail()  +(64*4), _MM_HINT_T0); // about 6-7 cpu cycle improvement (@ 10%)
        // is memory_order_release sufficent?
        finalTail_.store(tail_, std::memory_order_release);
    }

    char* get() { return ringBuff_.get(); }

    void reset()
    {
        for (int i = 0; i < ringBuffSize_; ++i)
            ringBuff_[i] = '5';

        memset(ringBuff_.get(), 0, ringBuffSize_);

        // eject log memory from cache
        for (int i = 0; i <ringBuffSize_; ++i)
            _mm_clflushopt(ringBuff_.get()+i);

        head_ = 0;
        tail_ = 0;
        lastHead_ = 0;
        lastTail_ = 0;
    }
};

SimpleRingBuff* pSRB{nullptr};
void segfault_sigaction(int signal, siginfo_t *si, void *arg)
{
    std::cout << "Caught segfault at address " << (intptr_t)si->si_addr << std::endl;
    std::cout << "Head = " << pSRB->getHead() << std::endl;
    std::cout << "Tail = " << pSRB->getTail() << std::endl;
    exit(0);
}

///////////////////////////////////////////////////////////////////////////////
uint64_t last{0};

template <std::size_t... Idx>
auto make_index_dispatcher(std::index_sequence<Idx...>) 
{
    return [] (auto&& f) { (f(std::integral_constant<std::size_t,Idx>{}), ...); };
}

template <std::size_t N>
auto make_index_dispatcher() 
{
    return make_index_dispatcher(std::make_index_sequence<N>{}); 
}

template <typename Tuple, typename Func>
void for_each(Tuple&& t, Func&& f) 
{
    constexpr auto n = std::tuple_size<std::decay_t<Tuple>>::value;
    auto dispatcher = make_index_dispatcher<n>();
    dispatcher([&f,&t](auto idx) { f(std::get<idx>(std::forward<Tuple>(t))); });
}

template <typename... Args>
uint64_t writeLog (SimpleRingBuff& srb)
{
    using Tuple_t = std::tuple<Args...>;
    Tuple_t *a = reinterpret_cast<Tuple_t*>(srb.pickConsume());

    // first element is timestamp, this is temporary to caluclate cycles between calls
    std::cout << std::get<0>(*a) - last << " ";
    last = std::get<0>(*a);

    if constexpr(sizeof...(Args) != 0)
    {
        // TODO? rework for printf style output using first argument of tuple as string?
        for_each(*a, [] (const auto& t) { std::cout << t << " ";});

        std::cout << " sizeof a = " << sizeof(std::tuple<Args...>);

        srb.consume(sizeof(Tuple_t));
    }

    return sizeof(Tuple_t);
}

///////////////////////////////////////////////////////////////////////////////
template <typename... Args>
uint64_t cbLog (SimpleRingBuff& srb)
{
    // advance past pointer to myself
    srb.consume(sizeof(void*));

    using Timestamp_t = int64_t;
    return writeLog<Timestamp_t, Args...>(srb);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
class Logger
{
public:
    constexpr static int dataStore{1024*1024*16};

private:
    SimpleRingBuff data;//[dataStore];

    int32_t lastFlush{0};
    int32_t flushMask{~0xF};
    uint32_t loggerCore_{0};

    std::unique_ptr<std::thread> logOut_;

public:
    Logger(uint32_t lc) : data(dataStore), loggerCore_(lc)
    {
        std::cout << "Data address = " << (int64_t)data.pickProduce() << std::endl;
        pSRB = &data;

        //sleep(1);
    }
    ~Logger()
    {
    }

    void start()
    {
        logOut_.reset(new std::thread(&Logger::printLogsT, this));
        setAffinity(*logOut_, loggerCore_);
    }

    ///////////////////////////////////////////////////////////////////////////////
#ifdef LOG_CLFLUSHOPT1
    //* origional
    template <typename... Args>
    void userLog (Args&&... args)
    {
        auto timeStamp = __rdtsc();
        auto f = cbLog<Args...>;

        // Is there a way to encode the function pointer into the tuple and still
        // properly extract later?
        [[maybe_unused]]decltype(f)* fp = new(data.pickProduce()) decltype(f)(f);
        data.produce(sizeof(void*));

        using Tuple_t = std::tuple<decltype(timeStamp), Args...>;

        // The beauty of placement new!
        Tuple_t *a = new(data.pickProduce()) Tuple_t(timeStamp, args...);
        data.produce(sizeof(*a));

        // use RIIA guard?
        data.cleanUpProduce();
    }
    // */
    //*
#elif LOG_CLFLUSHOPT2
    template <typename... Args>
    void userLog (Args&&... args)
    {
        auto timeStamp = __rdtsc();
        auto f = cbLog<Args...>;

        // Is there a way to encode the function pointer into the tuple and still
        // properly extract later?
        [[maybe_unused]]decltype(f)* fp = new(data.pickProduce()) decltype(f)(f);
        data.produce(sizeof(void*));

        using Tuple_t = std::tuple<decltype(timeStamp), Args...>;

        // added intermediate copy to be inline with NTS version
        // This is either as fast or faster than LOG_CLFLUSHOPT1.  Why?????
        //alignas(8) Tuple_t tmpTuple(timeStamp, args...);
        Tuple_t tmpTuple(timeStamp, args...);

        // The beauty of placement new!
        Tuple_t *a = new(data.pickProduce()) Tuple_t(tmpTuple);
        data.produce(sizeof(*a));

        // use RIIA guard?
        data.cleanUpProduce();
    }
    // */
#else
    //*
    template <typename... Args>
    void userLog (Args&&... args)
    {
        auto timeStamp = __rdtsc();
        auto f = cbLog<Args...>;

        // Is there a way to encode the function pointer into the tuple and still
        // properly extract later?
        [[maybe_unused]]decltype(f)* fp = new(data.pickProduce()) decltype(f)(f);
        data.produce(sizeof(void*));

        using Tuple_t = std::tuple<decltype(timeStamp), Args...>;

        alignas(8) Tuple_t tmpTuple(timeStamp, args...);

        char* store = data.pickProduce();

        using fakeIt_t = long long int;

        for (uint32_t i = 0; i < sizeof(tmpTuple)/8; ++i)
            _mm_stream_si64(reinterpret_cast<fakeIt_t*>(store)+i, *(reinterpret_cast<fakeIt_t*>(&tmpTuple)+i));

        _mm_sfence();

        data.produce(sizeof(tmpTuple));

        // use RIIA guard?
        data.cleanUpProduce();
    }
#endif

    // */
    ///////////////////////////////////////////////////////////////////////////////

    // refNumber and heapString bad code, but serve as examples
    uint32_t refNumber{0};
    template <typename... Args>
    uint64_t testLogs ( int32_t iter, Args... args)
    {
        uint16_t x = 987;
        char *heapString = new char[20];

        strncpy(heapString, "This is a heap string", 20);
        //warmup
        //for (int i = 0; i < 200; ++i)
        //    userLog(111, x, "BLEH", x, "fdsa", 222, 0.543, 333, 444, x, "ASDF");

        //data.reset();

        auto b = __rdtsc();
        for (int i = 0; i < iter; ++i)
        {
            refNumber = i;
            
            /*
            userLog(x, 222, 0.543, 333, 0.444, x, "ASDF", i+1, 555, x, 666, "funny", heapString, 'a', 'b', 'c', 'd'); 
            userLog(x, 222, 0.543, 333, 0.444, x, "ASDF", i+1, 555, x, 666, "funny", heapString, 'a', 'b', 'c', 'd'); 
            userLog(x, 222, 0.543, 333, 0.444, x, "ASDF", i+1, 555, x, 666, "funny", heapString, 'a', 'b', 'c', 'd'); 
            userLog(x, 222, 0.543, 333, 0.444, x, "ASDF", i+1, 555, x, 666, "funny", heapString, 'a', 'b', 'c', 'd'); 
            // */

            //*
            userLog(111, x, "BLEH ACH MEH", x, "fdsa", 222, 0.543, 333, 444, x, "ASDF", i, "jkl;");
            userLog("BLEH ACH MEH", x, "fdsa", 222, 0.543, 333, 444, x, "ASDF", i);
            // about 40 cpu cycles
            userLog(x, 222, 0.543, 333, 0.444, x, "ASDF", i+1, 555, x, 666, "funny", heapString, 'a', 'b', 'c', 'd'); 
            // about 22 cpu cycles
            userLog("args test ", args...); 
            // */

            /*
            userLog("too funny"); 
            userLog("too funny"); 
            userLog("too funny"); 
            userLog("too funny"); 
            // */
        }

        return (__rdtsc() - b)/4;
    }

    uint32_t prntIter_{0};

    void printLogs ()
    {
        using Fctn_t = uint64_t (*)(SimpleRingBuff& p);

        [[maybe_unused]] uint32_t iter{0};
        while (true)
        {
            //std::cout << "================" << std::endl;
            Fctn_t* fctn = (Fctn_t*)(data.pickConsume());

            // not the best way to detect end! Will correct later
            if (data.pickConsume() == nullptr)
            {
                //std::cout << "Flushing -- zero!" << std::endl;
                return;
            }

            std::cout   << ++iter << " Calling fctn " << (intptr_t)(void*)*fctn 
                        << " with data " << (intptr_t)data.pickConsume() << " prntIter = " << ++prntIter_ << std::endl;
            (*fctn)(data);
            // use RIIA guard?
            data.cleanUpConsume();
            std::cout << std::endl;
        }
    }

    // not working yet
    void printLogsT ()
    {
        [[maybe_unused]] uint32_t iter{0};
        while (true)
        {
            wait(3'000); // 1us
            //std::cout << "Print logs iter = " << ++iter << std::endl;
            printLogs();
        }
    }
};

int main ( int argc, char* argv[] )
{
    /*
    for (int i = 0; i < 100; ++i)
    {
        std::cout << "Waiting ..." << std::endl;
        wait(300'000'000); // 100ms
    }

    return 0;
    // */

    struct sigaction sa;

    memset(&sa, 0, sizeof(struct sigaction));
    sigemptyset(&sa.sa_mask);

    sa.sa_sigaction     = segfault_sigaction;
    sa.sa_flags         = SA_SIGINFO;

    sigaction(SIGSEGV, &sa, NULL);

    std::string pc{argv[1]};

    uint32_t core{0};
    uint32_t fastPathCore{0};
    uint32_t loggerCore{0};

    for (auto i : pc)
    {
        if (i == 'f')
        {
            std::cout << core << ":F ";
            fastPathCore = core;
        }
        else if (i == 'l')
        {
            std::cout << core << ":L ";
            loggerCore = core;
        }
        else
        {
            std::cout << core << ":N ";
        }

        ++core;
    }

    Logger log(loggerCore);
 
    constexpr int32_t numFctnPtr = 300;

    // arguments
    //
    constexpr int32_t numArgs = 20;
    uint8_t bytes1[numArgs]{0};
    uint16_t bytes2[numArgs]{1};
    uint32_t bytes4[numArgs]{2};
    uint64_t bytes8[numArgs]{3};
    float fbytes4[numArgs]{0.1};
    double dbytes8[numArgs]{0.2};

    const char* tstString1 = "This is an arg test";
    [[maybe_unused]]const char* tstString1a = "This is another arg test";



    auto fastPath = [&]()
    {
        auto time = log.testLogs(numFctnPtr);
        std::cerr << "Avg Log Time = " << time/numFctnPtr << std::endl;
        wait(300'000'000); // 100ms
        //log.printLogs();

        auto runningAvg = time/numFctnPtr;
        int i;
        int k = 0;
        for (i = 0; i < 30; ++i)
        {
            k = i % numArgs;
            time = log.testLogs(numFctnPtr
                    , bytes1[k]+i, bytes2[k]+i, bytes4[k]+i, bytes8[k]+i, fbytes4[k]+i, dbytes8[k]+i, tstString1[k] 
                    /*
                    , bytes1[k+1]+i, bytes2[k+1]+i, bytes4[k+1]+i, bytes8[k+1]+i, fbytes4[k+1]+i, dbytes8[k+1]+i, tstString1[k+1]
                    , bytes1[k+2]+i, bytes2[k+2]+i, bytes4[k+2]+i, bytes8[k+2]+i, fbytes4[k+2]+i, dbytes8[k+2]+i, tstString1[k+2]
                    , bytes1[k+3]+i, bytes2[k+3]+i, bytes4[k+3]+i, bytes8[k+3]+i, fbytes4[k+3]+i, dbytes8[k+3]+i, tstString1[k+3]
                    , bytes1[k+4]+i, bytes2[k+4]+i, bytes4[k+4]+i, bytes8[k+4]+i, fbytes4[k+4]+i, dbytes8[k+4]+i, tstString1[k+4]
                    , bytes1[k+5]+i, bytes2[k+5]+i, bytes4[k+5]+i, bytes8[k+5]+i, fbytes4[k+5]+i, dbytes8[k+5]+i, tstString1[k+5]
                    , bytes1[k+6]+i, bytes2[k+6]+i, bytes4[k+6]+i, bytes8[k+6]+i, fbytes4[k+6]+i, dbytes8[k+6]+i, tstString1[k+6]
                    , bytes1[k+7]+i, bytes2[k+7]+i, bytes4[k+7]+i, bytes8[k+7]+i, fbytes4[k+7]+i, dbytes8[k+7]+i, tstString1[k+7]
                    , bytes1[k+8]+i, bytes2[k+8]+i, bytes4[k+8]+i, bytes8[k+8]+i, fbytes4[k+8]+i, dbytes8[k+8]+i, tstString1[k+8]
                    , bytes1[k+9]+i, bytes2[k+9]+i, bytes4[k+9]+i, bytes8[k+9]+i, fbytes4[k+9]+i, dbytes8[k+9]+i, tstString1a[k+9]
                    // */
                    );
            std::cerr << "Avg Log Time = " << time/numFctnPtr << std::endl;
            //log.printLogs();
            runningAvg += time/numFctnPtr;
            wait(1'000'000); // 100ms
        }

        i += 1;

        std::cerr << "runningAvg = " << runningAvg / i << " i " << i << std::endl;
    };

    std::thread fp(fastPath);

    setAffinity(fp, fastPathCore);
    wait(300'000'000); // 100ms
    log.start();

    fp.join();

    return 0;
}
