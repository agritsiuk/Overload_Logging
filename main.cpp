///////////////////////////////////////////////////////////////////////////////
// Compile time options
//
// Used for different log encoding methodologies
// LOG_NTS - SSE non temporal store operations
// LOG_CLFLUSHOPT1, LOG_CLFLUSHOPT2, LOG_CLFLUSHOPT3 - use CLFLUSH[OPT]
//
// Used for measuring the impact of sfence for use with SSE NTS
// ADD_MARKER
//
// Used as an alternative flushing methodology requiring another thread
// to execute the flush command.
// FP_AUX
//
// FP_AUX should only be used with LOG_CLFLUSHOPT1, LOG_CLFLUSHOPT2 
// or LOG_CLFLUSHOPT3
//
//
// To run all threads on the same numa node on L29
// ./logging_poc  0.2.4.6.8.0f2l4a
// Cores 11, 13 and 15 will be used (15 only if FP_AUX is defined)
///////////////////////////////////////////////////////////////////////////////
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
// used to debug a few segfaults.
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
    SRBVec_t localSRBs;
    { // scope the lock guard
        std::lock_guard<std::mutex> lock(rbGuard_);
        localSRBs = ringBuffers_;
    }

    // TODO: FIXME
    // update for SRBVec_t to hold lastHead and lastTail or include with SRBs?
    // TESTING: only works with one RB 
    int32_t lastHead{0};
    int32_t lastTail{0};

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
        }

        //wait(300);
    }
}

}
///////////////////////////////////////////////////////////////////////////////


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

template <typename... T>
using TupleNR_t = std::tuple<typename std::decay<T>::type...>;
//using TupleNR_t = std::tuple<typename std::remove_reference<T>::type...>;

template <typename... Args>
struct alignas(8) Payload
{
    using Func_t = uint64_t (*)(SimpleRingBuff&);
    using Tuple_t = std::tuple<Args...>;

    Payload(Func_t f, Args&&... args) : func_(f), data(args...) {  }
    Func_t func_;
    //Tuple_t data;
    TupleNR_t<Args...> data; // construct tuple of values
};

// for calculating time between debug statements
uint64_t last{0};

template <typename... Args>
uint64_t writeLog (SimpleRingBuff& srb)
{
    using Payload_t = Payload<Args...>;

    Payload_t *a = reinterpret_cast<Payload_t*>(srb.pickConsume());

    // this should never happen?
    if (a == nullptr )//|| *(int*)&a->data == 0)
        return 0;

    if constexpr(sizeof...(Args) != 0)
    {
        std::cout << static_cast<float>(srb.getFreeSpace())/(1024*1024) << " ";
        // first element is timestamp, this is temporary to caluclate cycles between calls
        std::cout << std::get<0>(a->data) - last << " ";
        last = std::get<0>(a->data);

        // TODO? rework for printf style output using first argument of tuple as string?
        for_each(a->data, [] (const auto& t) { std::cout << t << " ";});

        std::cout << " sizeof Payload = " << sizeof(Payload_t) << " ";

        memset(a, 0, sizeof(Payload_t));

        std::cout << " after memset, " << (intptr_t)a->func_;
        srb.consume(sizeof(Payload_t));
    }

    return sizeof(Payload_t);
}

///////////////////////////////////////////////////////////////////////////////

template <typename... Args>
uint64_t cbLog (SimpleRingBuff& srb)
{
    using Timestamp_t = int64_t;
    return writeLog<Timestamp_t, Args...>(srb);


    //return writeLog<Timestamp_t, typename std::remove_reference<Args>::type...>(srb);
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

    uint32_t loggerCore_{0};
    uint32_t logMiss_{0};

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
    }

    void dbgPrint()
    {
        data.dbgPrint();
        std::cout << "Missed Logs " << logMiss_ << std::endl;
    }

    inline void sfence()
    {
#   ifdef ADD_MARKER
        _mm_sfence();
#   endif
    }

    ///////////////////////////////////////////////////////////////////////////////
#ifdef LOG_CLFLUSHOPT1
    template <typename... Args>
    void userLog (Args&&... args)
    {
        auto timeStamp = __rdtsc();
        using TimeStamp_t = decltype(timeStamp);
        using Payload_t = Payload<TimeStamp_t, Args...>;

        char* mem = data.pickProduce(sizeof(Payload_t));
        if (mem == nullptr)
        {
            ++logMiss_;
            return;
        }

        // The beauty of placement new!
        [[maybe_unused]]Payload_t* a = new(mem) Payload_t(
                  writeLog<TimeStamp_t, Args...>
                , std::forward<TimeStamp_t>(timeStamp)
                , std::forward<Args>(args)...);

        sfence();
        data.produce(sizeof(Payload_t));

        // use RIIA guard?
        data.cleanUpProduce();
    }
#elif LOG_CLFLUSHOPT2
    template <typename... Args>
    void userLog (Args&&... args)
    {
        auto timeStamp = __rdtsc();
        using TimeStamp_t = decltype(timeStamp);
        using Payload_t = Payload<TimeStamp_t, Args...>;

        char* mem = data.pickProduce(sizeof(Payload_t));
        if (mem == nullptr)
        {
            ++logMiss_;
            return;
        }

        Payload_t tmpPayload(
                  writeLog<TimeStamp_t, Args...>
                , std::forward<TimeStamp_t>(timeStamp)
                , std::forward<Args>(args)...);

        // The beauty of placement new!
        [[maybe_unused]]Payload_t* a = new(mem) Payload_t(tmpPayload);

        sfence();
        data.produce(sizeof(Payload_t));

        // use RIIA guard?
        data.cleanUpProduce();
    }
#elif LOG_CLFLUSHOPT3
    template <typename... Args>
    void userLog (Args&&... args)
    {
        auto timeStamp = __rdtsc();
        using TimeStamp_t = decltype(timeStamp);
        using Payload_t = Payload<TimeStamp_t, Args...>;

        char* mem = data.pickProduce(sizeof(Payload_t));
        if (mem == nullptr)
        {
            ++logMiss_;
            return;
        }

        Payload_t tmpPayload(
                  writeLog<TimeStamp_t, Args...>
                , std::forward<TimeStamp_t>(timeStamp)
                , std::forward<Args>(args)...);


        for (uint32_t i = 0; i < sizeof(Payload_t)/8; ++i)
            *(reinterpret_cast<uint64_t*>(mem)+i) = *(reinterpret_cast<uint64_t*>(&tmpPayload)+i);

        sfence();
        data.produce(sizeof(Payload_t));

        // use RIIA guard?
        data.cleanUpProduce();
    }
#elif LOG_NTS
    template <typename... Args>
    void userLog (Args&&... args)
    {
        auto timeStamp = __rdtsc();
        using TimeStamp_t = decltype(timeStamp);
        using Payload_t = Payload<TimeStamp_t, Args...>;

        char* mem = data.pickProduce(sizeof(Payload_t));
        if (mem == nullptr)
        {
            ++logMiss_;
            return;
        }

        Payload_t tmpPayload(
                  writeLog<TimeStamp_t, Args...>
                , std::forward<TimeStamp_t>(timeStamp)
                , std::forward<Args>(args)...);

        using FakeIt_t = long long int;

        for (uint32_t i = 0; i < sizeof(Payload_t)/8; ++i)
            _mm_stream_si64((reinterpret_cast<FakeIt_t*>(mem)+i), *(reinterpret_cast<FakeIt_t*>(&tmpPayload)+i));

        sfence();
        data.produce(sizeof(Payload_t));

        // use RIIA guard?
        data.cleanUpProduce();
    }
#endif
    ///////////////////////////////////////////////////////////////////////////////
    // refNumber and heapString bad code, but serve as examples
    uint32_t refNumber{0};
    template <typename... Args>
    uint64_t testLogs ( int32_t iter, Args&&... args)
    {
        char *heapString = new char[20];

        strncpy(heapString, "This is a heap string", 20);

        // need to work on timing (will probably do with new tapping POC, combine the two
        //uint64_t total{0};
        for (int i = 0; i < iter; ++i)
        {
            refNumber = i;
            //*
            //auto b = __rdtsc();
            //userLog("Diff paramters test, num parm ", sizeof...(Args)+4, i, std::ref(refNumber), heapString, args...);
            userLog("Diff paramters test, num parm ");//, sizeof...(Args)+4, i, std::ref(refNumber), heapString, args...);
            //auto t = __rdtsc() - b;
            //std::cerr << "\n" << t/1 << " TTT";
            // */
        }

        //total += __rdtsc() - b;
        return (0);
    }

    uint32_t prntIter_{0};

    void printLogs ()
    {
        using Fctn_t = uint64_t (*)(SimpleRingBuff& p);

        [[maybe_unused]] uint32_t iter{0};

        while (true)
        {
            void* mem = data.pickConsume();
            Fctn_t* fctn = (Fctn_t*)(mem);

            // Is there a better whay to detect nothing in queue?
            if (mem == 0 || *(int*)mem == 0)
            {
                //std::cout << "Flushing -- zero!" << std::endl;
                return;
            }

            (*fctn)(data);
            std::cout << std::endl;

            // clean up every log to keep cache pollution at minimum
            // use RIIA guard or part of Fctn_t?
            data.cleanUpConsume();
        }
    }

    void printLogsT ()
    {
        while (running_)
            printLogs();
    }
};

int main ( int argc, char* argv[] )
{
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
 
    constexpr int32_t numFctnPtr = 500;

    // arguments
    //
    constexpr int32_t numArgs = 20;
    [[maybe_unused]]uint8_t bytes1[numArgs]{0};
    [[maybe_unused]]uint16_t bytes2[numArgs]{1};
    [[maybe_unused]]uint32_t bytes4[numArgs]{2};
    [[maybe_unused]]uint64_t bytes8[numArgs]{3};
    [[maybe_unused]]float fbytes4[numArgs]{1.1};
    [[maybe_unused]]double dbytes8[numArgs]{1.2};

    [[maybe_unused]]const char* tstString1 = "This is an arg test";
    [[maybe_unused]]const char* tstString1a = "This is another arg test";

    [[maybe_unused]] uint16_t x = 987;

    auto fastPath = [&]()
    {
        auto runningAvg{0};
        int i;
        for (i = 0; i < 400; ++i)
        {
            [[maybe_unused]] auto time = log.testLogs(numFctnPtr
                    , 4ull, x, 2ull, 3ull, 4ull, 5ull, 6ull
                    /*
                    , bytes1[0]+i, bytes2[1]+i, bytes4[2]+i, bytes8[3]+i, fbytes4[4]+i, dbytes8[5]+i, tstString1
                    , bytes1[1]+i, bytes2[2]+i, bytes4[3]+i, bytes8[4]+i, fbytes4[5]+i, dbytes8[6]+i, tstString1
                    , bytes1[2]+i, bytes2[3]+i, bytes4[4]+i, bytes8[5]+i, fbytes4[6]+i, dbytes8[7]+i, tstString1
                    , bytes1[3]+i, bytes2[4]+i, bytes4[5]+i, bytes8[6]+i, fbytes4[7]+i, dbytes8[8]+i, tstString1
                    , bytes1[4]+i, bytes2[5]+i, bytes4[6]+i, bytes8[7]+i, fbytes4[8]+i, dbytes8[9]+i, tstString1
                    , bytes1[5]+i, bytes2[6]+i, bytes4[7]+i, bytes8[8]+i, fbytes4[9]+i, dbytes8[0]+i, tstString1
                    , bytes1[6]+i, bytes2[7]+i, bytes4[8]+i, bytes8[9]+i, fbytes4[0]+i, dbytes8[1]+i, tstString1
                    , bytes1[7]+i, bytes2[8]+i, bytes4[9]+i, bytes8[0]+i, fbytes4[1]+i, dbytes8[2]+i, tstString1
                    , bytes1[8]+i, bytes2[9]+i, bytes4[0]+i, bytes8[1]+i, fbytes4[2]+i, dbytes8[3]+i, tstString1
                    , bytes1[9]+i, bytes2[0]+i, bytes4[1]+i, bytes8[2]+i, fbytes4[3]+i, dbytes8[4]+i, tstString1a
                    // */
                    );
            //std::cerr << "Avg Log Time = " << time/numFctnPtr << std::endl;
            //runningAvg += time/numFctnPtr;
            //wait(9'000'000); // 100ms 
        }

        std::cerr << "runningAvg = " << runningAvg / i << " i " << i << std::endl;
    };
#ifdef FP_AUX
    std::thread auxFP(AuxFastPath::auxFastPathLoop);
    setAffinity(auxFP, auxFPCore);
#endif

    std::thread fp(fastPath);
    setAffinity(fp, fastPathCore);

    log.start();
    fp.join();
    log.stop();

#ifdef FP_AUX
    std::cout << "Stopping... " << std::endl;
    AuxFastPath::running = 0;
    auxFP.join();
#endif

    log.dbgPrint();

    return 0;
}
