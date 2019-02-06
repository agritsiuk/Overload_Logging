#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <functional>
#include <tuple>
#include <mutex>
#include <type_traits>

#include <assert.h>
#include <unistd.h>
#include <signal.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>

#include "srb.h"

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
    std::cerr << "Setting affinity for core " << cpuid << std::endl;
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

SimpleRingBuff* pSRB{nullptr};
void segfault_sigaction(int signal, siginfo_t *si, void *arg)
{
    std::cout << "***** Caught segfault at address " << (intptr_t)si->si_addr << std::endl;
    pSRB->dbgPrint();
    exit(0);
}

///////////////////////////////////////////////////////////////////////////////
namespace AuxFastPath
{
std::atomic<uint32_t> rbAdded{0};
std::atomic<uint32_t> running{1};

using SRBVec_t = std::vector<SimpleRingBuff*>;
SRBVec_t ringBuffers_;
std::mutex rbGuard_;

void addRingBuff (SimpleRingBuff& srb)
{
    { // scope the lock guard
    std::lock_guard<std::mutex> lock(rbGuard_);
    ringBuffers_.push_back(&srb);
    }

    rbAdded.store(1, std::memory_order_acquire);
}

void rbCleanup ( SimpleRingBuff& srb, int32_t& lastHead, int32_t& lastTail )
{
    lastHead = srb.flushProduce(lastHead);
    lastTail = srb.flushConsume(lastTail);
}

void auxFastPathLoop ( void )
{
    //*
    SRBVec_t localSRBs;
    { // scope the lock guard
        std::lock_guard<std::mutex> lock(rbGuard_);
        localSRBs = ringBuffers_;
    }

    // TESTING: only works with one RB 
    int32_t lastHead{0};
    int32_t lastTail{0};

    int32_t iter = 0;

    while (running.load(std::memory_order_relaxed))
    {
        if (rbAdded.load(std::memory_order_acquire))
        {
            std::lock_guard<std::mutex> lock(rbGuard_);
            localSRBs = ringBuffers_;
            rbAdded.store(0, std::memory_order_acquire);
        }

        for (auto i : localSRBs)
        {
            rbCleanup(*i, lastHead, lastTail);
            //if ( not (iter % 100))
            //    std::cout << "rbCleanup " << lastHead << " -- " << lastTail << std::endl;
        }

        //wait(300);
        ++iter;
    }
    // */
}

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
struct alignas(8) Payload
{
    using Func_t = uint64_t (*)(SimpleRingBuff&);
    using Tuple_t = std::tuple<Args...>;
        
    Payload(Func_t f, Args&&... args) : func_(f), data(args...) {  }
    Func_t func_;
    Tuple_t data;
};

template <typename... Args>
uint64_t writeLog (SimpleRingBuff& srb)
{
    using Payload_t = Payload<Args...>;

    Payload_t *a = reinterpret_cast<Payload_t*>(srb.pickConsume());

    // this should never happen?
    if (a == nullptr)
        return 0;

    if constexpr(sizeof...(Args) != 0)
    {
        // first element is timestamp, this is temporary to caluclate cycles between calls
        std::cout << std::get<0>(a->data) - last << " ";
        last = std::get<0>(a->data);

        // TODO? rework for printf style output using first argument of tuple as string?
        for_each(a->data, [] (const auto& t) { std::cout << t << " ";});

        std::cout << " sizeof Payload = " << sizeof(Payload_t);

        srb.consume(sizeof(Payload_t));
    }

    return sizeof(Payload_t);
}

///////////////////////////////////////////////////////////////////////////////

template <typename... Args>
uint64_t cbLog (SimpleRingBuff& srb)
{
    // advance past pointer to myself
    //srb.consume(sizeof(void*));

    using Timestamp_t = int64_t;
    return writeLog<Timestamp_t, Args...>(srb);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
class Logger
{
public:
    constexpr static int dataStore{1024*256*16};

private:
    SimpleRingBuff data;//[dataStore];

    int32_t lastFlush{0};
    int32_t flushMask{~0xF};
    uint32_t loggerCore_{0};

    std::unique_ptr<std::thread> logOut_;
    std::atomic_bool running_{false};
public:
    Logger(uint32_t lc) : data(dataStore), loggerCore_(lc)
    {
        std::cout << "Data address = " << (int64_t)data.pickProduce() << std::endl;
        pSRB = &data;

        AuxFastPath::addRingBuff(data);

        //sleep(1);
    }
    ~Logger()
    {
        if(logOut_ && logOut_->joinable())
        {
            running_ = false;
            logOut_->join();
        }
    }

    void start()
    {
        running_ = true;
        logOut_.reset(new std::thread(&Logger::printLogsT, this));
        setAffinity(*logOut_, loggerCore_);
    }

    void stop()
    {
        running_ = false;
        logOut_->join();
        data.dbgPrint();
    }

    ///////////////////////////////////////////////////////////////////////////////
#ifdef LOG_CLFLUSHOPT1
    //* origional


    template <typename... Args>
    void userLog (Args&&... args)
    {
        auto timeStamp = __rdtsc();
        using Payload_t = Payload<decltype(timeStamp), Args...>;

        void* mem = data.pickProduce(sizeof(Payload_t));
        if (mem == nullptr)
            return;

        // The beauty of placement new!
        [[maybe_unused]]Payload_t* a = new(mem) Payload_t(writeLog<decltype(timeStamp), Args...>, std::move(timeStamp), std::move(args)...);

#   ifdef ADD_MARKER
        _mm_sfence();
#   endif
        data.produce(sizeof(Payload_t));

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
        alignas(8) Tuple_t tmpTuple(timeStamp, args...);

        // The beauty of placement new!
        Tuple_t *a = new(data.pickProduce()) Tuple_t(tmpTuple);
#   ifdef ADD_MARKER
        _mm_sfence();
#   endif
        data.produce(sizeof(*a));

        // use RIIA guard?
        data.cleanUpProduce();
    }
    // */
#elif LOG_CLFLUSHOPT3
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
            *(reinterpret_cast<fakeIt_t*>(store)+i) = *(reinterpret_cast<fakeIt_t*>(&tmpTuple)+i);
#   ifdef ADD_MARKER
        _mm_sfence();
#   endif
        data.produce(sizeof(tmpTuple));

        // use RIIA guard?
        //data.cleanUpProduce();
    }
#else
    //*
    template <typename... Args>
    void userLog (Args&&... args)
    {
        auto timeStamp = __rdtsc();
        auto f = cbLog<Args...>;

        // Is there a way to encode the function pointer into the tuple and still
        // properly extract later?
        //[[maybe_unused]]decltype(f)* fp = new(data.pickProduce()) decltype(f)(f);
        _mm_stream_si64((long long int*)data.pickProduce(), (intptr_t)(f));
        data.produce(sizeof(void*));

        using Tuple_t = std::tuple<decltype(timeStamp), Args...>;

        alignas(8) Tuple_t tmpTuple(timeStamp, args...);

        char* store = data.pickProduce();
        assert(8 == (reinterpret_cast<intptr_t>(store) & 63));

        using fakeIt_t = long long int;

        for (uint32_t i = 0; i < sizeof(tmpTuple)/8; ++i)
            _mm_stream_si64(reinterpret_cast<fakeIt_t*>(store)+i, *(reinterpret_cast<fakeIt_t*>(&tmpTuple)+i));

#   ifdef ADD_MARKER
        _mm_sfence();
#   endif

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
        [[maybe_unused]] uint16_t x = 987;
        // char *heapString = new char[20];

        // strncpy(heapString, "This is a heap string", 20);
        //warmup
        //for (int i = 0; i < 200; ++i)
        //    userLog(111, x, "BLEH", x, "fdsa", 222, 0.543, 333, 444, x, "ASDF");

        //data.reset();

        uint64_t total{0};
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
            userLog(1ull, 2ull, 3ull, 4ull, 5ull, 6ull);
            //wait(300'000);
            //userLog(111, x, "BLEH ACH MEH", x, "fdsa", 222, 0.543, 333, 444, x, "ASDF", i, "jkl;");
            //userLog("BLEH ACH MEH", x, "fdsa", 222, 0.543, 333, 444, x, "ASDF", i);
            // about 40 cpu cycles
            //userLog(x, 222, 0.543, 333, 0.444, x, "ASDF", i+1, 555, x, 666, "funny", heapString, 'a', 'b', 'c', 'd');
            // about 22 cpu cycles
            //userLog("args test ", args...);
            // */

            /*
            userLog("too funny"); 
            userLog("too funny"); 
            userLog("too funny"); 
            userLog("too funny"); 
            // */
        }

        total += __rdtsc() - b;
        return (total/1);
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

            // Is there a better whay to detect nothing in queue?
            if (data.pickConsume() == nullptr || fctn == 0)
            {
                //std::cout << "Flushing -- zero!" << std::endl;
                return;
            }

            //std::cout   << ++iter << " Calling fctn " << (intptr_t)(void*)*fctn
            //            << " with data " << (intptr_t)data.pickConsume() << " prntIter = " << ++prntIter_ << std::endl;
            (*fctn)(data);
            std::cout << std::endl;

            // clean up every log to keep cache pollution at minimum
            // use RIIA guard or part of Fctn_t?
            data.cleanUpConsume();
        }
    }

    // not working yet
    void printLogsT ()
    {
        [[maybe_unused]] uint32_t iter{0};
        while (running_)
        {
            //wait(3'000); // 1us
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
    [[maybe_unused]]uint32_t auxFPCore{0};

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
        else if (i == 'a')
        {
            std::cout << core << ":A ";
            auxFPCore = core;
        }
        else
        {
            std::cout << core << ":N ";
        }

        ++core;
    }

    Logger log(loggerCore);
 
    constexpr int32_t numFctnPtr = 1'000;

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
        //wait(300'000'000); // 100ms
        //log.printLogs();

        auto runningAvg = time/numFctnPtr;
        int i;
        for (i = 0; i < 2'000; ++i)
        {
            time = log.testLogs(numFctnPtr
                    /*
                    , bytes1[0]+i, bytes2[0]+i, bytes4[0]+i, bytes8[0]+i, fbytes4[0]+i, dbytes8[0]+i, tstString1[0]
                    , bytes1[1]+i, bytes2[1]+i, bytes4[1]+i, bytes8[1]+i, fbytes4[1]+i, dbytes8[1]+i, tstString1[1]
                    , bytes1[2]+i, bytes2[2]+i, bytes4[2]+i, bytes8[2]+i, fbytes4[2]+i, dbytes8[2]+i, tstString1[2]
                    , bytes1[3]+i, bytes2[3]+i, bytes4[3]+i, bytes8[3]+i, fbytes4[3]+i, dbytes8[3]+i, tstString1[3]
                    , bytes1[4]+i, bytes2[4]+i, bytes4[4]+i, bytes8[4]+i, fbytes4[4]+i, dbytes8[4]+i, tstString1[4]
                    , bytes1[5]+i, bytes2[5]+i, bytes4[5]+i, bytes8[5]+i, fbytes4[5]+i, dbytes8[5]+i, tstString1[5]
                    , bytes1[6]+i, bytes2[6]+i, bytes4[6]+i, bytes8[6]+i, fbytes4[6]+i, dbytes8[6]+i, tstString1[6]
                    , bytes1[7]+i, bytes2[7]+i, bytes4[7]+i, bytes8[7]+i, fbytes4[7]+i, dbytes8[7]+i, tstString1[7]
                    , bytes1[8]+i, bytes2[8]+i, bytes4[8]+i, bytes8[8]+i, fbytes4[8]+i, dbytes8[8]+i, tstString1[8]
                    , bytes1[9]+i, bytes2[9]+i, bytes4[9]+i, bytes8[9]+i, fbytes4[9]+i, dbytes8[9]+i, tstString1a[9]
                    // */
                    );
            std::cerr << "Avg Log Time = " << time/numFctnPtr << std::endl;
            //log.printLogs();
            runningAvg += time/numFctnPtr;
            // FIXME
            // avoid overrun (will fix later)
            // Solution is to have pick methods provide an argument of how much space to use
            wait(3'000'000); // 100ms
        }

        i += 1;

        std::cerr << "runningAvg = " << runningAvg / i << " i " << i << std::endl;
    };
#ifdef FP_AUX
    std::thread auxFP(AuxFastPath::auxFastPathLoop);
    setAffinity(auxFP, auxFPCore);
#endif



    std::thread fp(fastPath);
    setAffinity(fp, fastPathCore);
    //wait(300'000'000); // 100ms
    log.start();

    fp.join();
    log.stop();

#ifdef FP_AUX
    std::cout << "Stopping... " << std::endl;
    AuxFastPath::running = 0;
    auxFP.join();
#endif

    return 0;
}
