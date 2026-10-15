//----------------------------------------------------------------------------------------------------------------------
///
/// \file       benchmark_lap.cpp
/// \brief      Тестирование задач о назначениях
///

#define BOOST_TEST_MODULE benchmark_lap

//#define PRINTTOTXT
#define USECOLORGRAPHS

// #define CYCLE_LONG 1000
// #define CYCLE_SHORT 100

// #define CYCLE_LONG 400
// #define CYCLE_SHORT 20

#define CYCLE_LONG 100
#define CYCLE_SHORT 20

// #define CYCLE_LONG 1
// #define CYCLE_SHORT 1

#include <boost/test/included/unit_test.hpp>
#include <vector>
#include <set>
#include <map>
#include <armadillo>
#include <random>
#include <fstream>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <thread>
#include <chrono>

#include "spm/timing/timing.hpp"
#include "spm/lap/lap.hpp"
#include "spm/lap/sparse.hpp"
#include "spm/lap/murty.hpp"

#ifdef MATPLOTLIB
    #include <matplotlibcpp.h>
#endif

bool show = false;//true;//

//----------------------------------------------------------------------------------------------------------------------
// Тестовые задачи:
BOOST_AUTO_TEST_CASE( accordance )
{
    const int n = 128;
    arma::mat mat_JVCdense( n, n, arma::fill::randn );
    arma::mat mat_Mack( n, n, arma::fill::randn );
    arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
    arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
    double lapcostJVCdense = 0.0, lapcostMack = 0.0;//, lapcostHungarian = 0.0;
    int cycle_count = 10000;
    for( int cycle = 0; cycle < cycle_count; cycle++ ) {
        std::cout << "accordance cycle = " << cycle << "/" << cycle_count << std::endl;
        arma::arma_rng::set_seed( cycle );
        mat_JVCdense.randn();
        arma::arma_rng::set_seed( cycle );
        mat_Mack.randn();

        double infValue = 1e7;
        double resolution = 1e-7;
        SPM::LAP::JVCdense( mat_JVCdense, n, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actualJVCdense, lapcostJVCdense );

        SPM::LAP::Mack( mat_Mack, n, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actualMack, lapcostMack );

        BOOST_CHECK_EQUAL( lapcostJVCdense, lapcostMack );

        for( int i = 0; i < n; i++ ) {
            BOOST_CHECK_EQUAL( ( std::abs( actualJVCdense(i) - actualMack(i) ) < 1e-5 ), true );
        }
    }
}

BOOST_AUTO_TEST_CASE( cycling_jvc )
{
    // Тест проверки на зацикливание метода JVC
    const int K = 96;
    const int L = 32;
    const int N = ( K + L );
    double psi_empty = -1e6;
    double resolution = 1e-7;
    arma::mat assigncost( N, N, arma::fill::zeros ); // Полная матрица ценности (включая пустые назначения до размера k+l)
    arma::mat filledcost( K, L, arma::fill::zeros ); // Матрица ценности (без пустых назначений до размера k+l )
    arma::ivec result = arma::ivec( N, arma::fill::zeros );
    double lapcost;
    int cycle_count = 10000;

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

    for( int cycle = 0; cycle < cycle_count; cycle++ ) {
        std::cout << "test_LAP_JV_cycling cycle = " << ( cycle + 1 ) << "/" << cycle_count << std::endl;
        assigncost.fill( psi_empty );
        filledcost.randu();
        // Проредим матрицу filledcost: в случайных местах
        double levelCut = 0.5; // Порог прореживания
        for( int i = 0; i < K; i++ ) {
            for( int j = 0; j < L; j++ ) {
                double randomVal = random_0_1( gen );// double randomVal = drand( 0.0, 1.0 );
                if( randomVal < levelCut ) {
                    filledcost(i,j) = psi_empty; // Проредим матрицу
                }
            }
        }
        assigncost.submat( 0, 0, ( K - 1 ), ( L - 1 ) ) = filledcost;
        SPM::LAP::JVCdense( assigncost, N, SPM::LAP::TSearchParam::SP_Max, std::abs( psi_empty ), resolution,
            result, lapcost );
    }
}

BOOST_AUTO_TEST_CASE( cycling_murty )
{
    // Тест проверки на зацикливание метода Murty
    const int K = 96;
    const int L = 32;
    const int N = ( K + L );
    double psi_empty = -1e6;
    double resolution = 1e-7;
    arma::mat assigncost( N, N, arma::fill::zeros ); // Полная матрица ценности (включая пустые назначения до размера k+l)
    arma::mat filledcost( K, L, arma::fill::zeros ); // Матрица ценности (без пустых назначений до размера k+l )
    arma::ivec result = arma::ivec( N, arma::fill::zeros );
    int cycle_count = 10000;

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

    int Kbest = 5; // Число лучших решений

    for( int cycle = 0; cycle < cycle_count; cycle++ ) {
        std::cout << "test_Murty_cycling cycle = " << ( cycle + 1 ) << "/" << cycle_count << std::endl;
        assigncost.fill( psi_empty );
        filledcost.randu();
        // Проредим матрицу filledcost: в случайных местах
        double levelCut = 0.5; // Порог прореживания
        for( int i = 0; i < K; i++ ) {
            for( int j = 0; j < L; j++ ) {
                double randomVal = random_0_1( gen );// double randomVal = drand( 0.0, 1.0 );
                if( randomVal < levelCut ) {
                    filledcost(i,j) = psi_empty; // Проредим матрицу
                }
            }
        }
        assigncost.submat( 0, 0, ( K - 1 ), ( L - 1 ) ) = filledcost;
        SPM::Sparse::CMatrixCSR csr;
        SPM::Sparse::MatrixDenseToCSR( assigncost, csr );
        SPM::LAP::Murty murty( csr, SPM::LAP::TSearchParam::SP_Max, std::abs( psi_empty ), resolution );
        SPM::LAP::MurtySolution out;
        for( int i = 0; i < Kbest; i++ ) {
            murty.findNext( out );
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

double levelCut = 0.02; // Порог прореживания

const double width = 50;//100;//
const double height = 20;//20;//
const double dpi = 300;// dpi (variable)

std::map<std::string, double> keywords{
    { "left", 0.13 },
    { "bottom", 0.2 },
    { "right", 0.95 },
    { "top", 0.95 },
    { "wspace", 0.4 },
    { "hspace", 0.4 }
};

BOOST_AUTO_TEST_CASE( denseSmall )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;

    bool print = true; //false; //
    int cycle_count = CYCLE_LONG;

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;

    const double in2mm = 25.4;// mm (fixed)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, мс" );
#endif
    std::vector<int> dimensionXlim;
    int startPow = 2;
    int endPow = 6;
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
        if( i < endPow ) {
            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );
            int val_1 = value + ( ( valueNext - value ) / 4 );
            int val_2 = value + ( ( valueNext - value ) / 2 );
            int val_3 = value + 3 * ( ( valueNext - value ) / 4 );
            if( val_1 != 5 ) {
                dimensionXlim.push_back( val_1 );
            }
            dimensionXlim.push_back( val_2 );
            if( val_3 != 7 ) {
                dimensionXlim.push_back( val_3 );
            }
        }
    }
    std::vector<double> dimensionLongLong( dimensionXlim.begin(), dimensionXlim.end() );
    std::vector<double> dimensionShort( dimensionXlim.begin(), dimensionXlim.end() );
    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLongLong },
        { "Mack", dimensionLongLong },
        { "Hungarian", dimensionShort }
    };
    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };
    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
#ifdef USECOLORGRAPHS
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
#else
        { "JVCdense", { { "color", "black" }, {"marker", "x"}, { "linestyle", "-" }, { "linewidth", "1.5" }, { "label", "JVCdense" } } },
        { "Mack", { { "color", "black" }, {"marker", "^"}, { "linestyle", "-." }, { "linewidth", "1.5" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "black" }, {"marker", "s"}, { "linestyle", ":" }, { "linewidth", "1.5" }, { "label", "Hungarian" } } }
#endif
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );
        double lapcostJVCdense = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0;

        std::map<std::string, SPM::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPM::Timing::CTimeKeeper() },
            { "Mack", SPM::Timing::CTimeKeeper() },
            { "Hungarian", SPM::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }
            /////////////////////
//            mat_JVCdense.print();
            /////////////////////

            mat_Mack = mat_JVCdense;
            mat_Hungarian = mat_JVCdense;

            double maxcost = 1e7;

            timer.at( "JVCdense" ).StartTimer();
            SPM::LAP::JVCdense( mat_JVCdense, n, sp, maxcost, resolution, actualJVCdense, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            SPM::LAP::Mack( mat_Mack, n, sp, maxcost, resolution, actualMack, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPM::LAP::Hungarian( mat_Hungarian, n, sp, maxcost, resolution, actualHungarian, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            bool eq_lapcostJVCdense_lapcostMack = std::abs( lapcostJVCdense - lapcostMack ) < 1e-5;
            bool eq_lapcostJVCdense_lapcostHungarian = false;

            if( n <= boundN ) {
                eq_lapcostJVCdense_lapcostHungarian = std::abs( lapcostJVCdense - lapcostHungarian ) < 1e-5;
            }

            if( !eq_lapcostJVCdense_lapcostMack ||
                ( ( n <= boundN ) && ( !eq_lapcostJVCdense_lapcostHungarian ) ) )
            {
                counter_lapcost++;
            }

            double eps = 1e-7;
            bool eq_actualJVCdense_actualMack = arma::approx_equal( actualJVCdense, actualMack, "absdiff", eps );
            bool eq_actualJVCdense_actualHungarian = false;
            if( n <= boundN ) {
                eq_actualJVCdense_actualHungarian = arma::approx_equal( actualJVCdense, actualHungarian, "absdiff", eps );
            }
            if( !eq_actualJVCdense_actualMack ||
                ( ( n <= boundN ) && ( !eq_actualJVCdense_actualHungarian ) ) )
            {
                counter_assign++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( n > boundN ) && ( t.first == "Hungarian" ) ) {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;

#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        plt::plot(
            dimensionDouble.at( t.first ),
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::subplots_adjust( keywords );
    std::vector<int> dimensionXlimDiv;
    for( std::size_t i = 0; i < dimensionXlim.size(); i++ ) {
        if( ( i % 2 ) == 0 ) {
            dimensionXlimDiv.push_back( dimensionXlim[i] );
        }
    }
    plt::xticks( dimensionXlimDiv );
    plt::xlim( 8, dimensionXlim.back() );//plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::ylim( -0.2, 3.0 );
    plt::legend();
    plt::save( "denseSmall.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif

#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "denseSmall.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            os << (t.second)[i] << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( denseSmallSeqExtr )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;

    bool print = true;
    int cycle_count = CYCLE_LONG;

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;

    const double in2mm = 25.4;// mm (fixed)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, мс" );
#endif
    std::vector<int> dimensionXlim;
    int startPow = 2;
    int endPow = 6;
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
        if( i < endPow ) {
            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );
            int val_1 = value + ( ( valueNext - value ) / 4 );
            int val_2 = value + ( ( valueNext - value ) / 2 );
            int val_3 = value + 3 * ( ( valueNext - value ) / 4 );
            if( val_1 != 5 ) {
                dimensionXlim.push_back( val_1 );
            }
            dimensionXlim.push_back( val_2 );
            if( val_3 != 7 ) {
                dimensionXlim.push_back( val_3 );
            }
        }
    }
    std::vector<double> dimensionLongLong( dimensionXlim.begin(), dimensionXlim.end() );
    std::vector<double> dimensionShort( dimensionXlim.begin(), dimensionXlim.end() );
    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "SeqExtr", dimensionLongLong },
        { "JVCdense", dimensionLongLong },
        { "Mack", dimensionLongLong },
        { "Hungarian", dimensionShort }
    };
    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "SeqExtr", {} },
        { "JVCdense", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };
    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
#ifdef USECOLORGRAPHS
        { "SeqExtr", { { "color", "orange" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtr" } } },
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
#else
        { "SeqExtr", { { "color", "black" }, {"marker", "v"}, { "linestyle", "-." }, { "linewidth", "1.5" }, { "label", "SeqExtr" } } },
        { "JVCdense", { { "color", "black" }, {"marker", "x"}, { "linestyle", "-" }, { "linewidth", "1.5" }, { "label", "JVCdense" } } },
        { "Mack", { { "color", "black" }, {"marker", "^"}, { "linestyle", "-." }, { "linewidth", "1.5" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "black" }, {"marker", "s"}, { "linestyle", ":" }, { "linewidth", "1.5" }, { "label", "Hungarian" } } }
#endif
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {
        arma::mat mat_SeqExtr( n, n, arma::fill::randn );
        arma::mat mat_JVCdense( n, n, arma::fill::randn );
        arma::mat mat_Mack( n, n, arma::fill::randn );
        arma::mat mat_Hungarian( n, n, arma::fill::randn );

        arma::ivec actualSeqExtr = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );
        double lapcostSeqExtr = 0.0, lapcostJVCdense = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0;

        std::map<std::string, SPM::Timing::CTimeKeeper> timer = {
            { "SeqExtr", SPM::Timing::CTimeKeeper() },
            { "JVCdense", SPM::Timing::CTimeKeeper() },
            { "Mack", SPM::Timing::CTimeKeeper() },
            { "Hungarian", SPM::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }
            /////////////////////
//            mat_JVCdense.print();
            /////////////////////

            mat_SeqExtr = mat_JVCdense;
            mat_Mack = mat_JVCdense;
            mat_Hungarian = mat_JVCdense;

            double maxcost = 1e7;

            timer.at( "JVCdense" ).StartTimer();
            SPM::LAP::JVCdense( mat_JVCdense, n, sp, maxcost, resolution, actualJVCdense, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "SeqExtr" ).StartTimer();
            SPM::LAP::SequentalExtremum( mat_SeqExtr, sp, maxcost, resolution, actualSeqExtr, lapcostSeqExtr );
            timer.at( "SeqExtr" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            SPM::LAP::Mack( mat_Mack, n, sp, maxcost, resolution, actualMack, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPM::LAP::Hungarian( mat_Hungarian, n, sp, maxcost, resolution, actualHungarian, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            bool eq_lapcostJVCdense_lapcostMack = std::abs( lapcostJVCdense - lapcostMack ) < 1e-5;
            bool eq_lapcostJVCdense_lapcostHungarian = false;

            if( n <= boundN ) {
                eq_lapcostJVCdense_lapcostHungarian = std::abs( lapcostJVCdense - lapcostHungarian ) < 1e-5;
            }

            if( !eq_lapcostJVCdense_lapcostMack ||
                ( ( n <= boundN ) && ( !eq_lapcostJVCdense_lapcostHungarian ) ) )
            {
                counter_lapcost++;
            }

            double eps = 1e-7;
            bool eq_actualJVCdense_actualMack = arma::approx_equal( actualJVCdense, actualMack, "absdiff", eps );
            bool eq_actualJVCdense_actualHungarian = false;
            if( n <= boundN ) {
                eq_actualJVCdense_actualHungarian = arma::approx_equal( actualJVCdense, actualHungarian, "absdiff", eps );
            }
            if( !eq_actualJVCdense_actualMack ||
                ( ( n <= boundN ) && ( !eq_actualJVCdense_actualHungarian ) ) )
            {
                counter_assign++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( n > boundN ) && ( t.first == "Hungarian" ) ) {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        plt::plot(
            dimensionDouble.at( t.first ),
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::subplots_adjust( keywords );

    std::vector<int> dimensionXlimDiv;
    for( std::size_t i = 0; i < dimensionXlim.size(); i++ ) {
        if( ( i % 2 ) == 0 ) {
            dimensionXlimDiv.push_back( dimensionXlim[i] );
        }
    }
    plt::xticks( dimensionXlimDiv );
    plt::xlim( 8, dimensionXlim.back() );//plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::ylim( -0.2, 3.0 );
    plt::legend();
    plt::save( "denseSmallSeqExtr.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif

#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "denseSmallSeqExtr.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            os << (t.second)[i] << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( sparseSmall )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;

    bool print = true;
    int cycle_count = CYCLE_LONG;

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    std::uniform_real_distribution<double> random_double_probability( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;

    const double in2mm = 25.4;// mm (fixed)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, мс" );
#endif

    std::vector<int> dimensionXlim;
    int startPow = 2;
    int endPow = 6;
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
        if( i < endPow ) {
            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );
            int val_1 = value + ( ( valueNext - value ) / 4 );
            int val_2 = value + ( ( valueNext - value ) / 2 );
            int val_3 = value + 3 * ( ( valueNext - value ) / 4 );
            if( val_1 != 5 ) {
                dimensionXlim.push_back( val_1 );
            }
            dimensionXlim.push_back( val_2 );
            if( val_3 != 7 ) {
                dimensionXlim.push_back( val_3 );
            }
        }
    }
    std::vector<double> dimensionLongLong( dimensionXlim.begin(), dimensionXlim.end() );
    std::vector<double> dimensionShort( dimensionXlim.begin(), dimensionXlim.end() );

    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLongLong },
        { "JVCsparse", dimensionLongLong },
        { "Mack", dimensionLongLong },
        { "Hungarian", dimensionShort }
    };
    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "JVCsparse", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };
    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
#ifdef USECOLORGRAPHS
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
#else
        { "JVCdense", { { "color", "black" }, {"marker", "x"}, { "linestyle", "-" }, { "linewidth", "1.5" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "black" }, {"marker", "o"}, { "linestyle", "--" }, { "linewidth", "1.5" }, { "label", "JVCsparse" } } },
        { "Mack", { { "color", "black" }, {"marker", "^"}, { "linestyle", "-." }, { "linewidth", "1.5" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "black" }, {"marker", "s"}, { "linestyle", ":" }, { "linewidth", "1.5" }, { "label", "Hungarian" } } }
#endif
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPM::Sparse::CMatrixCSR mat_JVCsparse;
        SPM::Sparse::CMatrixCSR mat_JVCsparse2N;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCdense = 0.0, lapcostJVCsparse = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0;

        std::map<std::string, SPM::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPM::Timing::CTimeKeeper() },
            { "JVCsparse", SPM::Timing::CTimeKeeper() },
            { "Mack", SPM::Timing::CTimeKeeper() },
            { "Hungarian", SPM::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }
            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;

            double infValue = 1e7;
            double bigValue = -1e6;
            double halfBigValue = -1e5;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = bigValue;
                            mat_JVCdenseForSparse(i, j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPM::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::ones );
            mat_JVCdense2N *= bigValue;
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= bigValue;
            arma::vec emptyDenseDiag = arma::vec( n, arma::fill::ones );
            emptyDenseDiag *= halfBigValue;
            emptyDense.diag() = emptyDenseDiag;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
            emptySparse *= halfBigValue;

            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
            verbotenSparse *= halfBigValue;

            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT
//            mat_JVCdense2N.print( "mat_JVCdense2N" );
//            mat_JVCdenseForSparse2N.print( "mat_JVCdenseForSparse2N" );
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPM::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );

            // Запуск решений с замерами времени выполнения
            timer.at( "JVCdense" ).StartTimer();
            SPM::LAP::JVCdense( mat_JVCdense2N, 2*n, sp, infValue, resolution, actualJVCdense2N, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPM::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            timer.at( "Mack" ).StartTimer();
            SPM::LAP::Mack( mat_Mack2N, 2*n, sp, infValue, resolution, actualMack2N, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPM::LAP::Hungarian( mat_Hungarian2N, 2*n, sp, infValue, resolution, actualHungarian2N, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
            std::vector< std::pair<int, int> > solutionMack;
            std::vector< std::pair<int, int> > solutionHungarian;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCdense2N(i) < n ) {
                    solutionJVCdense.push_back( std::make_pair( i, actualJVCdense2N(i) ) );
                }
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
                if( actualMack2N(i) < n ) {
                    solutionMack.push_back( std::make_pair( i, actualMack2N(i) ) );
                }
                if( n <= boundN ) { // Для венгерки
                    if( actualHungarian2N(i) < n ) {
                        solutionHungarian.push_back( std::make_pair( i, actualHungarian2N(i) ) );
                    }
                }
            }

            if( ( solutionJVCdense.size() != solutionJVCsparse.size() ) ||
                ( solutionJVCdense.size() != solutionMack.size() ) ||
                ( ( n <= boundN ) && ( solutionJVCdense.size() != solutionHungarian.size() ) ) )
            {
                counter_assign++;
            } else {
                int size = solutionJVCdense.size();
                for( int i = 0; i < size; i++ ) {
                    if( ( solutionJVCdense[i].first != solutionJVCsparse[i].first ) ||
                        ( solutionJVCdense[i].second != solutionJVCsparse[i].second ) ||
                        ( solutionJVCdense[i].first != solutionMack[i].first ) ||
                        ( solutionJVCdense[i].second != solutionMack[i].second ) ||
                        ( ( n <= boundN ) && ( ( solutionJVCdense[i].first != solutionHungarian[i].first ) ||
                        ( solutionJVCdense[i].second != solutionHungarian[i].second ) ) ) )
                    {
                        counter_assign++;
                        break;
                    }
                }
            }

            // Оценим LAPCOST - суммарную стоимость назначения

            double totalcostJVCdense = 0.0;
            for( auto &elem : solutionJVCdense ) {
                totalcostJVCdense += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostJVCsparse = 0.0;
            for( auto &elem : solutionJVCsparse ) {
                totalcostJVCsparse += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostHungarian = 0.0;
            for( auto &elem : solutionHungarian ) {
                totalcostHungarian += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostMack = 0.0;
            for( auto &elem : solutionMack ) {
                totalcostMack+= mat_JVCdense( elem.first, elem.second );
            }

            bool eq_totalcostJVCdense_totalcostMack = std::abs( totalcostJVCdense - totalcostMack ) < 1e-5;
            bool eq_totalcostJVCdense_totalcostJVCsparse = std::abs( totalcostJVCdense - totalcostJVCsparse ) < 1e-5;
            bool eq_totalcostJVCdense_totalcostHungarian = false;
            if( n <= boundN ) {
                eq_totalcostJVCdense_totalcostHungarian = std::abs( totalcostJVCdense - totalcostHungarian ) < 1e-5;
            }

            if( !eq_totalcostJVCdense_totalcostMack ||
                !eq_totalcostJVCdense_totalcostJVCsparse ||
                ( ( n <= boundN ) && ( !eq_totalcostJVCdense_totalcostHungarian ) ) )
            {
                counter_lapcost++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( n > boundN ) && ( t.first == "Hungarian" ) ) {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        std::vector<double> dim = dimensionDouble.at( t.first );
        for( auto &ii : dim ) {
            ii *= 2;
        }
        plt::plot(
//            dimensionDouble.at( t.first ),
            dim,
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::subplots_adjust( keywords );
    std::vector<int> dimensionXlimDiv;
    for( std::size_t i = 0; i < dimensionXlim.size(); i++ ) {
        if( ( i % 2 ) == 0 ) {
            dimensionXlimDiv.push_back( dimensionXlim[i] );
        }
    }
    plt::xticks( dimensionXlimDiv );
    plt::xlim( 8, dimensionXlim.back() );//plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::ylim( -0.2, 3.0 );
    plt::legend();
    plt::save( "sparseSmall.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "sparseSmall.ods", std::ofstream::out );
    if( print ) {
        std::cout << "PRINTTOTXT..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( sparseSmallSeqExtr )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;

    bool print = true; //false; //
    int cycle_count = CYCLE_LONG;

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    std::uniform_real_distribution<double> random_double_probability( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;

    const double in2mm = 25.4;// mm (fixed)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, мс" );
#endif

    std::vector<int> dimensionXlim;
    int startPow = 2;
    int endPow = 6;
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
        if( i < endPow ) {
            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );

            int val_1 = value + ( ( valueNext - value ) / 4 );
            int val_2 = value + ( ( valueNext - value ) / 2 );
            int val_3 = value + 3 * ( ( valueNext - value ) / 4 );
            if( val_1 != 5 ) {
                dimensionXlim.push_back( val_1 );
            }
            dimensionXlim.push_back( val_2 );
            if( val_3 != 7 ) {
                dimensionXlim.push_back( val_3 );
            }
        }
    }
    std::vector<double> dimensionLongLong( dimensionXlim.begin(), dimensionXlim.end() );
    std::vector<double> dimensionShort( dimensionXlim.begin(), dimensionXlim.end() );

    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLongLong },
        { "JVCsparse", dimensionLongLong },
        { "SeqExtr", dimensionLongLong },
        { "SeqExtrCOO", dimensionLongLong },
        { "Mack", dimensionLongLong },
        { "Hungarian", dimensionShort }
    };
    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "JVCsparse", {} },
        { "SeqExtr", {} },
        { "SeqExtrCOO", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };
    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
#ifdef USECOLORGRAPHS
        { "SeqExtr", { { "color", "orange" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtr" } } },
        { "SeqExtrCOO", { { "color", "yellow" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtrCOO" } } },
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
#else
        { "SeqExtr", { { "color", "black" }, {"marker", "v"}, { "linestyle", "-." }, { "linewidth", "1.5" }, { "label", "SeqExtr" } } },
        { "SeqExtrCOO", { { "color", "black" }, {"marker", "d"}, { "linestyle", "-." }, { "linewidth", "1" }, { "label", "SeqExtrCOO" } } },
        { "JVCdense", { { "color", "black" }, {"marker", "x"}, { "linestyle", "-" }, { "linewidth", "1.5" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "black" }, {"marker", "o"}, { "linestyle", "--" }, { "linewidth", "1.5" }, { "label", "JVCsparse" } } },
        { "Mack", { { "color", "black" }, {"marker", "^"}, { "linestyle", "-." }, { "linewidth", "1.5" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "black" }, {"marker", "s"}, { "linestyle", ":" }, { "linewidth", "1.5" }, { "label", "Hungarian" } } }
#endif
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPM::Sparse::CMatrixCSR mat_JVCsparse;
        SPM::Sparse::CMatrixCSR mat_JVCsparse2N;
        SPM::Sparse::CMatrixCOO mat_COOsparse;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_SeqExtr2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtr2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtrCOO2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCdense = 0.0, lapcostJVCsparse = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0,
            lapcostSeqExtr = 0.0, lapcostSeqExtrCOO = 0.0;

        std::map<std::string, SPM::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPM::Timing::CTimeKeeper() },
            { "JVCsparse", SPM::Timing::CTimeKeeper() },
            { "SeqExtr", SPM::Timing::CTimeKeeper() },
            { "SeqExtrCOO", SPM::Timing::CTimeKeeper() },
            { "Mack", SPM::Timing::CTimeKeeper() },
            { "Hungarian", SPM::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }
            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;

            double infValue = 1e7;
            double bigValue = -1e6;
            double halfBigValue = -1e5;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = bigValue;
                            mat_JVCdenseForSparse(i,j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPM::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::ones );
            mat_JVCdense2N *= bigValue;
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= bigValue;
            arma::vec emptyDenseDiag = arma::vec( n, arma::fill::ones );
            emptyDenseDiag *= halfBigValue;
            emptyDense.diag() = emptyDenseDiag;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
            emptySparse *= halfBigValue;

            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
            verbotenSparse *= halfBigValue;

            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT
//            mat_JVCdense.print( "mat_JVCdense" );
//            mat_JVCdense2N.print( "mat_JVCdense2N" );
//            mat_JVCdenseForSparse2N.print( "mat_JVCdenseForSparse2N" );
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPM::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            mat_SeqExtr2N = mat_JVCdense2N;
            SPM::Sparse::MatrixDenseToCOO( mat_JVCdenseForSparse2N, mat_COOsparse );

            // Запуск решений с замерами времени выполнения
            timer.at( "JVCdense" ).StartTimer();
            SPM::LAP::JVCdense( mat_JVCdense2N, 2*n, sp, infValue, resolution, actualJVCdense2N, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPM::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            timer.at( "SeqExtr" ).StartTimer();
            SPM::LAP::SequentalExtremum( mat_SeqExtr2N, sp, infValue, resolution, actualSeqExtr2N, lapcostSeqExtr );
            timer.at( "SeqExtr" ).EndTimer();

            timer.at( "SeqExtrCOO" ).StartTimer();
            SPM::LAP::SequentalExtremum( mat_COOsparse, sp, infValue, resolution, actualSeqExtrCOO2N, lapcostSeqExtrCOO );
            timer.at( "SeqExtrCOO" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            SPM::LAP::Mack( mat_Mack2N, 2*n, sp, infValue, resolution, actualMack2N, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPM::LAP::Hungarian( mat_Hungarian2N, 2*n, sp, infValue, resolution, actualHungarian2N, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
            std::vector< std::pair<int, int> > solutionMack;
            std::vector< std::pair<int, int> > solutionHungarian;
            std::vector< std::pair<int, int> > solutionSeqExtr;
            std::vector< std::pair<int, int> > solutionSeqExtrCOO;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCdense2N(i) < n ) {
                    solutionJVCdense.push_back( std::make_pair( i, actualJVCdense2N(i) ) );
                }
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
                if( actualMack2N(i) < n ) {
                    solutionMack.push_back( std::make_pair( i, actualMack2N(i) ) );
                }
                if( actualSeqExtr2N(i) < n ) {
                    solutionSeqExtr.push_back( std::make_pair( i, actualSeqExtr2N(i) ) );
                }
                if( actualSeqExtrCOO2N(i) < n ) {
                    solutionSeqExtrCOO.push_back( std::make_pair( i, actualSeqExtrCOO2N(i) ) );
                }
                if( n <= boundN ) { // Для венгерки
                    if( actualHungarian2N(i) < n ) {
                        solutionHungarian.push_back( std::make_pair( i, actualHungarian2N(i) ) );
                    }
                }
            }

            if( ( solutionJVCdense.size() != solutionJVCsparse.size() ) ||
                ( solutionJVCdense.size() != solutionMack.size() ) ||
                ( ( n <= boundN ) && ( solutionJVCdense.size() != solutionHungarian.size() ) ) )
            {
                counter_assign++;
            } else {
                int size = solutionJVCdense.size();
                for( int i = 0; i < size; i++ ) {
                    if( ( solutionJVCdense[i].first != solutionJVCsparse[i].first ) ||
                        ( solutionJVCdense[i].second != solutionJVCsparse[i].second ) ||
                        ( solutionJVCdense[i].first != solutionMack[i].first ) ||
                        ( solutionJVCdense[i].second != solutionMack[i].second ) ||
                        ( ( n <= boundN ) && ( ( solutionJVCdense[i].first != solutionHungarian[i].first ) ||
                        ( solutionJVCdense[i].second != solutionHungarian[i].second ) ) ) )
                    {
                        counter_assign++;
                        break;
                    }
                }
            }

            // Оценим LAPCOST - суммарную стоимость назначения

            double totalcostJVCdense = 0.0;
            for( auto &elem : solutionJVCdense ) {
                totalcostJVCdense += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostJVCsparse = 0.0;
            for( auto &elem : solutionJVCsparse ) {
                totalcostJVCsparse += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostHungarian = 0.0;
            for( auto &elem : solutionHungarian ) {
                totalcostHungarian += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostMack = 0.0;
            for( auto &elem : solutionMack ) {
                totalcostMack+= mat_JVCdense( elem.first, elem.second );
            }

            bool eq_totalcostJVCdense_totalcostMack = std::abs( totalcostJVCdense - totalcostMack ) < 1e-5;
            bool eq_totalcostJVCdense_totalcostJVCsparse = std::abs( totalcostJVCdense - totalcostJVCsparse ) < 1e-5;
            bool eq_totalcostJVCdense_totalcostHungarian = false;
            if( n <= boundN ) {
                eq_totalcostJVCdense_totalcostHungarian = std::abs( totalcostJVCdense - totalcostHungarian ) < 1e-5;
            }

            if( !eq_totalcostJVCdense_totalcostMack ||
                !eq_totalcostJVCdense_totalcostJVCsparse ||
                ( ( n <= boundN ) && ( !eq_totalcostJVCdense_totalcostHungarian ) ) )
            {
                counter_lapcost++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( n > boundN ) && ( t.first == "Hungarian" ) ) {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
//    for( auto &t : timeOfMethod ) {
//        plt::plot(
//            dimensionDouble.at( t.first ),
//            ( t.second ), // Y
//            estimated_keywords.at( t.first )
//            );
//    }
    for( auto &t : timeOfMethod ) {
        std::vector<double> dim = dimensionDouble.at( t.first );
        for( auto &ii : dim ) {
            ii *= 2;
        }
        plt::plot(
//            dimensionDouble.at( t.first ),
            dim,
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::subplots_adjust( keywords );
    std::vector<int> dimensionXlimDiv;
    for( std::size_t i = 0; i < dimensionXlim.size(); i++ ) {
        if( ( i % 2 ) == 0 ) {
            dimensionXlimDiv.push_back( dimensionXlim[i] );
        }
    }
    plt::xticks( dimensionXlimDiv );
    plt::xlim( 8, dimensionXlim.back() );//plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::ylim( -0.2, 3.0 );
    plt::legend();
    plt::save( "sparseSmallSeqExtr.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "sparseSmallSeqExtr.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( denseLarge )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;

    bool print = true; //false; //
    int cycle_count = CYCLE_SHORT;

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;

    const double in2mm = 25.4;// mm (fixed)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, мс" );
#endif

    std::vector<double> dimensionLongLong;
    std::vector<double> dimensionLong;
    std::vector<double> dimensionShort;

    std::vector<int> dimensionXlim;
    int startPow = 7;
    int endPow = 10;
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
        if( i < endPow ) {
            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );
            int val_1 = value + ( ( valueNext - value ) / 4 );
            int val_2 = value + ( ( valueNext - value ) / 2 );
            int val_3 = value + 3 * ( ( valueNext - value ) / 4 );
            if( val_1 != 5 ) {
                dimensionXlim.push_back( val_1 );
            }
            dimensionXlim.push_back( val_2 );
            if( val_3 != 7 ) {
                dimensionXlim.push_back( val_3 );
            }
        }
    }
    for( auto &el : dimensionXlim ) {
        double val = static_cast<double>( el );
        dimensionLongLong.push_back( val );
        if( val < 513 ) {
            dimensionLong.push_back( val );
        }
        if( val < 300 ) {
            dimensionShort.push_back( val );
        }
    }
    int boundN = dimensionShort.back();
    int boundN2 = dimensionLong.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLongLong },
        { "Mack", dimensionLong },
        { "Hungarian", dimensionShort }
    };
    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };
    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
#ifdef USECOLORGRAPHS
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
#else
        { "JVCdense", { { "color", "black" }, {"marker", "x"}, { "linestyle", "-" }, { "linewidth", "1.5" }, { "label", "JVCdense" } } },
        { "Mack", { { "color", "black" }, {"marker", "^"}, { "linestyle", "-." }, { "linewidth", "1.5" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "black" }, {"marker", "s"}, { "linestyle", ":" }, { "linewidth", "1.5" }, { "label", "Hungarian" } } }
#endif
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {

        arma::mat mat_JVCdense( n, n, arma::fill::randn );
        arma::mat mat_Mack( n, n, arma::fill::randn );
        arma::mat mat_Hungarian( n, n, arma::fill::randn );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );
        double lapcostJVCdense = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0;

        std::map<std::string, SPM::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPM::Timing::CTimeKeeper() },
            { "Mack", SPM::Timing::CTimeKeeper() },
            { "Hungarian", SPM::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }
            /////////////////////
//            mat_JVCdense.print();
            /////////////////////

            mat_Mack = mat_JVCdense;
            mat_Hungarian = mat_JVCdense;

            double maxcost = 1e7;

            timer.at( "JVCdense" ).StartTimer();
            SPM::LAP::JVCdense( mat_JVCdense, n, sp, maxcost, resolution, actualJVCdense, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            if( n <= boundN2 ) {
                timer.at( "Mack" ).StartTimer();
                SPM::LAP::Mack( mat_Mack, n, sp, maxcost, resolution, actualMack, lapcostMack );
                timer.at( "Mack" ).EndTimer();
            }

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPM::LAP::Hungarian( mat_Hungarian, n, sp, maxcost, resolution, actualHungarian, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            bool eq_lapcostJVCdense_lapcostMack = false;
            if( n <= boundN2 ) {
                eq_lapcostJVCdense_lapcostMack = std::abs( lapcostJVCdense - lapcostMack ) < 1e-5;
            }

            bool eq_lapcostJVCdense_lapcostHungarian = false;
            if( n <= boundN ) {
                eq_lapcostJVCdense_lapcostHungarian = std::abs( lapcostJVCdense - lapcostHungarian ) < 1e-5;
            }

            if( ( ( n <= boundN2 ) && ( !eq_lapcostJVCdense_lapcostMack ) ) ||
                ( ( n <= boundN )  && ( !eq_lapcostJVCdense_lapcostHungarian ) ) )
            {
                counter_lapcost++;
            }

            double eps = 1e-7;

            bool eq_actualJVCdense_actualMack = false;
            if( n <= boundN2 ) {
                eq_actualJVCdense_actualMack = arma::approx_equal( actualJVCdense, actualMack, "absdiff", eps );
            }

            bool eq_actualJVCdense_actualHungarian = false;
            if( n <= boundN ) {
                eq_actualJVCdense_actualHungarian = arma::approx_equal( actualJVCdense, actualHungarian, "absdiff", eps );
            }

            if( ( ( n <= boundN2 ) && ( !eq_actualJVCdense_actualMack ) ) ||
                ( ( n <= boundN )  && ( !eq_actualJVCdense_actualHungarian ) ) )
            {
                counter_assign++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( ( n <= boundN2 ) && ( t.first == "Mack" ) ) ||
                ( ( n <= boundN )  && ( t.first == "Hungarian" ) ) ||
                ( t.first == "JVCdense" )
                )
            {
                timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
            }
        }
    }
    std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
    std::cout << "counter_assign=" << counter_assign << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        plt::plot(
            dimensionDouble.at( t.first ),
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::subplots_adjust( keywords );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    std::vector<int> dimensionXlimDiv;
    for( std::size_t i = 0; i < dimensionXlim.size(); i++ ) {
        if( ( i % 2 ) == 0 ) {
            dimensionXlimDiv.push_back( dimensionXlim[i] );
        }
    }
    plt::xticks( dimensionXlimDiv );
    plt::ylim( -100.0, 1500.0 );//3000.0 );
    plt::legend();
    plt::save( "denseLarge.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif

#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "denseLarge.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

/*
BOOST_AUTO_TEST_CASE( sparseLarge )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;

    bool print = true; //false; //
    int cycle_count = CYCLE_SHORT;

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    std::uniform_real_distribution<double> random_double_probability( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;

    const double in2mm = 25.4;// mm (fixed)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, мс" );
#endif
    std::vector<double> dimensionLong;
    std::vector<int> dimensionXlim;
    int startPow = 7;
    int endPow = 10;
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
        if( i < endPow ) {
            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );
            int val_1 = value + ( ( valueNext - value ) / 4 );
            int val_2 = value + ( ( valueNext - value ) / 2 );
            int val_3 = value + 3 * ( ( valueNext - value ) / 4 );
            if( val_1 != 5 ) {
                dimensionXlim.push_back( val_1 );
            }
            dimensionXlim.push_back( val_2 );
            if( val_3 != 7 ) {
                dimensionXlim.push_back( val_3 );
            }
        }
    }
    for( auto &el : dimensionXlim ) {
        double val = static_cast<double>( el );
        dimensionLong.push_back( val );
    }

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLong },
        { "JVCsparse", dimensionLong }
    };
    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "JVCsparse", {} }
    };
    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } }
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPM::Sparse::CMatrixCSR mat_JVCsparse;
        SPM::Sparse::CMatrixCSR mat_JVCsparse2N;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCdense = 0.0, lapcostJVCsparse = 0.0;//, lapcostMack = 0.0, lapcostHungarian = 0.0;

        std::map<std::string, SPM::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPM::Timing::CTimeKeeper() },
            { "JVCsparse", SPM::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }
            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;

            double infValue = 1e7;
            double bigValue = -1e6;
            double halfBigValue = -1e5;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = bigValue;
                            mat_JVCdenseForSparse(i,j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPM::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::ones );
            mat_JVCdense2N *= bigValue;
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= bigValue;
            arma::vec emptyDenseDiag = arma::vec( n, arma::fill::ones );
            emptyDenseDiag *= halfBigValue;
            emptyDense.diag() = emptyDenseDiag;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
            emptySparse *= halfBigValue;

            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
            verbotenSparse *= halfBigValue;

            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT
//            mat_JVCdense2N.print();
//            mat_JVCdenseForSparse2N.print();
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPM::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;

            // Запуск решений с замерами времени выполнения
            timer.at( "JVCdense" ).StartTimer();
            SPM::LAP::JVCdense( mat_JVCdense2N, 2*n, sp, infValue, resolution, actualJVCdense2N, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPM::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
            std::vector< std::pair<int, int> > solutionMack;
            std::vector< std::pair<int, int> > solutionHungarian;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCdense2N(i) < n ) {
                    solutionJVCdense.push_back( std::make_pair( i, actualJVCdense2N(i) ) );
                }
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
            }

            if( ( solutionJVCdense.size() != solutionJVCsparse.size() ) )
            {
                counter_assign++;
            } else {
                int size = solutionJVCdense.size();
                for( int i = 0; i < size; i++ ) {
                    if( ( solutionJVCdense[i].first != solutionJVCsparse[i].first ) ||
                        ( solutionJVCdense[i].second != solutionJVCsparse[i].second ) )
                    {
                        counter_assign++;
                        break;
                    }
                }
            }

            // Оценим LAPCOST - суммарную стоимость назначения
            double totalcostJVCdense = 0.0;
            for( auto &elem : solutionJVCdense ) {
                totalcostJVCdense += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostJVCsparse = 0.0;
            for( auto &elem : solutionJVCsparse ) {
                totalcostJVCsparse += mat_JVCdense( elem.first, elem.second );
            }
            bool eq_totalcostJVCdense_totalcostJVCsparse = std::abs( totalcostJVCdense - totalcostJVCsparse ) < 1e-5;

            if( !eq_totalcostJVCdense_totalcostJVCsparse )
            {
                counter_lapcost++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
//    for( auto &t : timeOfMethod ) {
//        plt::plot(
//            dimensionDouble.at( t.first ),
//            ( t.second ), // Y
//            estimated_keywords.at( t.first )
//            );
//    }

    for( auto &t : timeOfMethod ) {
        std::vector<double> dim = dimensionDouble.at( t.first );
        for( auto &ii : dim ) {
            ii *= 2;
        }
        plt::plot(
//            dimensionDouble.at( t.first ),
            dim,
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }

    plt::grid( true );
    plt::subplots_adjust( keywords );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    std::vector<int> dimensionXlimDiv;
    for( std::size_t i = 0; i < dimensionXlim.size(); i++ ) {
        if( ( i % 2 ) == 0 ) {
            dimensionXlimDiv.push_back( dimensionXlim[i] );
        }
    }
    plt::xticks( dimensionXlimDiv );
    plt::ylim( -50.0, 1000.0 );
    plt::legend();
    plt::save( "sparseLarge.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "sparseLarge.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}
*/
BOOST_AUTO_TEST_CASE( sparseLarge )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;

    bool print = true; //false; //
    int cycle_count = CYCLE_SHORT;

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    std::uniform_real_distribution<double> random_double_probability( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;

    const double in2mm = 25.4;// mm (fixed)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, мс" );
#endif
    std::vector<double> dimensionLongLong;
    std::vector<double> dimensionLong;
    std::vector<double> dimensionShort;
    std::vector<int> dimensionXlim;
    int startPow = 6;
    int endPow = 10;//10
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
        if( i < endPow ) {
            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );
            int val_1 = value + ( ( valueNext - value ) / 4 );
            int val_2 = value + ( ( valueNext - value ) / 2 );
            int val_3 = value + 3 * ( ( valueNext - value ) / 4 );
            if( val_1 != 5 ) {
                dimensionXlim.push_back( val_1 );
            }
            dimensionXlim.push_back( val_2 );
            if( val_3 != 7 ) {
                dimensionXlim.push_back( val_3 );
            }
        }
    }
    for( auto &el : dimensionXlim ) {
        double val = static_cast<double>( el );
        dimensionLongLong.push_back( val );
        if( val < ( 512 * 0.5 ) + 1 ) {
            dimensionLong.push_back( val );
        }
        if( val < ( 256 * 0.5 ) + 1 ) {
            dimensionShort.push_back( val );
        }
    }
    int boundN = dimensionShort.back();
    int boundN2 = dimensionLong.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLongLong },
        { "JVCsparse", dimensionLongLong },
        { "Mack", dimensionLong },
        { "Hungarian", dimensionShort }
    };
    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "JVCsparse", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };
    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
#ifdef USECOLORGRAPHS
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
#else
        { "JVCdense", { { "color", "black" }, {"marker", "x"}, { "linestyle", "-" }, { "linewidth", "1.5" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "black" }, {"marker", "o"}, { "linestyle", "--" }, { "linewidth", "1.5" }, { "label", "JVCsparse" } } },
        { "Mack", { { "color", "black" }, {"marker", "^"}, { "linestyle", "-." }, { "linewidth", "1.5" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "black" }, {"marker", "s"}, { "linestyle", ":" }, { "linewidth", "1.5" }, { "label", "Hungarian" } } }
#endif
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPM::Sparse::CMatrixCSR mat_JVCsparse;
        SPM::Sparse::CMatrixCSR mat_JVCsparse2N;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCdense = 0.0, lapcostJVCsparse = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0;

        std::map<std::string, SPM::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPM::Timing::CTimeKeeper() },
            { "JVCsparse", SPM::Timing::CTimeKeeper() },
            { "Mack", SPM::Timing::CTimeKeeper() },
            { "Hungarian", SPM::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }
            mat_JVCdenseForSparse = mat_JVCdense;
            mat_Mack = mat_JVCdense;
            mat_Hungarian = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;

            double infValue = 1e7;
            double bigValue = -1e6;
            double halfBigValue = -1e5;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = bigValue;
                            mat_JVCdenseForSparse(i,j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPM::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::ones );
            mat_JVCdense2N *= bigValue;
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= bigValue;
            arma::vec emptyDenseDiag = arma::vec( n, arma::fill::ones );
            emptyDenseDiag *= halfBigValue;
            emptyDense.diag() = emptyDenseDiag;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
            emptySparse *= halfBigValue;

            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
            verbotenSparse *= halfBigValue;

            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT
//            mat_JVCdense2N.print();
//            mat_JVCdenseForSparse2N.print();
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPM::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;

            // Запуск решений с замерами времени выполнения
            timer.at( "JVCdense" ).StartTimer();
            SPM::LAP::JVCdense( mat_JVCdense2N, 2*n, sp, infValue, resolution, actualJVCdense2N, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPM::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            if( n <= boundN2 ) {
                timer.at( "Mack" ).StartTimer();
                SPM::LAP::Mack( mat_Mack2N, 2*n, sp, infValue, resolution, actualMack2N, lapcostMack );
                timer.at( "Mack" ).EndTimer();
            }

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPM::LAP::Hungarian( mat_Hungarian2N, 2*n, sp, infValue, resolution, actualHungarian2N, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
            std::vector< std::pair<int, int> > solutionMack;
            std::vector< std::pair<int, int> > solutionHungarian;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCdense2N(i) < n ) {
                    solutionJVCdense.push_back( std::make_pair( i, actualJVCdense2N(i) ) );
                }
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
                if( n <= boundN2 ) {
                    if( actualMack2N(i) < n ) {
                        solutionMack.push_back( std::make_pair( i, actualMack2N(i) ) );
                    }
                }
                if( n <= boundN ) {
                    if( actualHungarian2N(i) < n ) {
                        solutionHungarian.push_back( std::make_pair( i, actualHungarian2N(i) ) );
                    }
                }
            }

            if( ( solutionJVCdense.size() != solutionJVCsparse.size() ) )
            {
                counter_assign++;
            } else {
                int size = solutionJVCdense.size();
                for( int i = 0; i < size; i++ ) {
                    if( ( solutionJVCdense[i].first != solutionJVCsparse[i].first ) ||
                        ( solutionJVCdense[i].second != solutionJVCsparse[i].second ) )
                    {
                        counter_assign++;
                        break;
                    }
                }
            }

            // Оценим LAPCOST - суммарную стоимость назначения
            double totalcostJVCdense = 0.0;
            for( auto &elem : solutionJVCdense ) {
                totalcostJVCdense += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostJVCsparse = 0.0;
            for( auto &elem : solutionJVCsparse ) {
                totalcostJVCsparse += mat_JVCdense( elem.first, elem.second );
            }
            bool eq_totalcostJVCdense_totalcostJVCsparse = std::abs( totalcostJVCdense - totalcostJVCsparse ) < 1e-5;

            if( !eq_totalcostJVCdense_totalcostJVCsparse )
            {
                counter_lapcost++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

//        for( auto &t : timer ) {
//            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
//        }

        for( auto &t : timer ) {
            if( ( ( n > boundN ) && ( t.first == "Hungarian" ) ) ||
                ( ( n > boundN2 ) && ( t.first == "Mack" ) ) )
            {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }

        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
//    for( auto &t : timeOfMethod ) {
//        plt::plot(
//            dimensionDouble.at( t.first ),
//            ( t.second ), // Y
//            estimated_keywords.at( t.first )
//            );
//    }

    for( auto &t : timeOfMethod ) {
        std::vector<double> dim = dimensionDouble.at( t.first );
        for( auto &ii : dim ) {
            ii *= 2;
        }
        plt::plot(
//            dimensionDouble.at( t.first ),
            dim,
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }

    plt::grid( true );
    plt::subplots_adjust( keywords );
    std::vector<int> dimensionXlimDiv;
    for( std::size_t i = 0; i < dimensionXlim.size(); i++ ) {
        if( ( i % 2 ) == 0 ) {
            dimensionXlimDiv.push_back( dimensionXlim[i] );
        }
    }
    plt::xticks( dimensionXlimDiv );
    plt::xlim( dimensionXlim.front() * 2, dimensionXlim.back() );
    plt::ylim( -100.0, 1500.0 );//3000.0 );
    plt::legend();
    plt::save( "sparseLarge.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "sparseLarge.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( sparseLargeSeqExtr )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;

    bool print = true; //false; //
    int cycle_count = CYCLE_SHORT;

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    std::uniform_real_distribution<double> random_double_probability( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;

    const double in2mm = 25.4;// mm (fixed)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, мс" );
#endif

    std::vector<double> dimensionLong;
    std::vector<int> dimensionXlim;
    int startPow = 7;
    int endPow = 9;
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
        if( i < endPow ) {
            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );

            int val_1 = value + ( ( valueNext - value ) / 4 );
            int val_2 = value + ( ( valueNext - value ) / 2 );
            int val_3 = value + 3 * ( ( valueNext - value ) / 4 );
            if( val_1 != 5 ) {
                dimensionXlim.push_back( val_1 );
            }
            dimensionXlim.push_back( val_2 );
            if( val_3 != 7 ) {
                dimensionXlim.push_back( val_3 );
            }
        }
    }
    for( auto &el : dimensionXlim ) {
        double val = static_cast<double>( el );
        dimensionLong.push_back( val );
    }

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLong },
        { "JVCsparse", dimensionLong },
        { "SeqExtrCOO", dimensionLong }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "JVCsparse", {} },
        { "SeqExtrCOO", {} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
#ifdef USECOLORGRAPHS
        { "SeqExtrCOO", { { "color", "yellow" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtrCOO" } } },
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } }
#else
        { "SeqExtrCOO", { { "color", "black" }, {"marker", "d"}, { "linestyle", "-." }, { "linewidth", "1" }, { "label", "SeqExtrCOO" } } },
        { "JVCdense", { { "color", "black" }, {"marker", "x"}, { "linestyle", "-" }, { "linewidth", "1.5" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "black" }, {"marker", "o"}, { "linestyle", "--" }, { "linewidth", "1.5" }, { "label", "JVCsparse" } } }
#endif
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPM::Sparse::CMatrixCSR mat_JVCsparse;
        SPM::Sparse::CMatrixCSR mat_JVCsparse2N;
        SPM::Sparse::CMatrixCOO mat_COOsparse;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_SeqExtr2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtr2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtrCOO2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCdense = 0.0, lapcostJVCsparse = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0,
            lapcostSeqExtr = 0.0, lapcostSeqExtrCOO = 0.0;
        ///

        std::map<std::string, SPM::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPM::Timing::CTimeKeeper() },
            { "JVCsparse", SPM::Timing::CTimeKeeper() },
            { "SeqExtrCOO", SPM::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }
            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;

            double infValue = 1e7;
            double bigValue = -1e6;
            double halfBigValue = -1e5;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = bigValue;
                            mat_JVCdenseForSparse(i,j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPM::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::ones );
            mat_JVCdense2N *= bigValue;
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= bigValue;
            arma::vec emptyDenseDiag = arma::vec( n, arma::fill::ones );
            emptyDenseDiag *= halfBigValue;
            emptyDense.diag() = emptyDenseDiag;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
            emptySparse *= halfBigValue;

            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
            verbotenSparse *= halfBigValue;

            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT
//            mat_JVCdense2N.print();
//            mat_JVCdenseForSparse2N.print();
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPM::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            mat_SeqExtr2N = mat_JVCdense2N;
            SPM::Sparse::MatrixDenseToCOO( mat_JVCdenseForSparse2N, mat_COOsparse );

            // Запуск решений с замерами времени выполнения
            timer.at( "JVCdense" ).StartTimer();
            SPM::LAP::JVCdense( mat_JVCdense2N, 2*n, sp, infValue, resolution, actualJVCdense2N, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "SeqExtrCOO" ).StartTimer();
            SPM::LAP::SequentalExtremum( mat_COOsparse, sp, infValue, resolution, actualSeqExtrCOO2N, lapcostSeqExtrCOO );
            timer.at( "SeqExtrCOO" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPM::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
            std::vector< std::pair<int, int> > solutionSeqExtrCOO;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCdense2N(i) < n ) {
                    solutionJVCdense.push_back( std::make_pair( i, actualJVCdense2N(i) ) );
                }
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
            }

            if( ( solutionJVCdense.size() != solutionJVCsparse.size() ) )
            {
                counter_assign++;
            } else {
                int size = solutionJVCdense.size();
                for( int i = 0; i < size; i++ ) {
                    if( ( solutionJVCdense[i].first != solutionJVCsparse[i].first ) ||
                        ( solutionJVCdense[i].second != solutionJVCsparse[i].second ) )
                    {
                        counter_assign++;
                        break;
                    }
                }
            }

            // Оценим LAPCOST - суммарную стоимость назначения
            double totalcostJVCdense = 0.0;
            for( auto &elem : solutionJVCdense ) {
                totalcostJVCdense += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostJVCsparse = 0.0;
            for( auto &elem : solutionJVCsparse ) {
                totalcostJVCsparse += mat_JVCdense( elem.first, elem.second );
            }
            bool eq_totalcostJVCdense_totalcostJVCsparse = std::abs( totalcostJVCdense - totalcostJVCsparse ) < 1e-5;

            if( !eq_totalcostJVCdense_totalcostJVCsparse )
            {
                counter_lapcost++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
//    for( auto &t : timeOfMethod ) {
//        plt::plot(
//            dimensionDouble.at( t.first ),
//            ( t.second ), // Y
//            estimated_keywords.at( t.first )
//            );
//    }
    for( auto &t : timeOfMethod ) {
        std::vector<double> dim = dimensionDouble.at( t.first );
        for( auto &ii : dim ) {
            ii *= 2;
        }
        plt::plot(
//            dimensionDouble.at( t.first ),
            dim,
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }

    plt::grid( true );
    plt::subplots_adjust( keywords );
    std::vector<int> dimensionXlimDiv;
    for( std::size_t i = 0; i < dimensionXlim.size(); i++ ) {
        if( ( i % 2 ) == 0 ) {
            dimensionXlimDiv.push_back( dimensionXlim[i] );
        }
    }
    plt::xticks( dimensionXlimDiv );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::ylim( -50.0, 1000.0 );
    plt::legend();
    plt::save( "sparseLargeSeqExtr.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "sparseLargeSeqExtr.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( sparseLargeExtra )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;
    bool print = true; //false; //
    int cycle_count = 1;//CYCLE_SHORT;//

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    std::uniform_real_distribution<double> random_double_probability( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;

    const double in2mm = 25.4;// mm (fixed)
    const double mm2px = dpi / in2mm;//
    double width2 = width * 1.25;
    size_t pixels_width = std::round( width2 * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, с" );
#endif

    std::vector<double> dimensionLong;
    std::vector<int> dimensionXlim;
    int startPow = 9;
    int endPow = 14;
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
        if( i < endPow ) {
            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );
//            int val_1 = value + ( ( valueNext - value ) / 4 );
            int val_2 = value + ( ( valueNext - value ) / 2 );
//            int val_3 = value + 3 * ( ( valueNext - value ) / 4 );

//            if( val_1 != 5 ) {
//                dimensionXlim.push_back( val_1 );
//            }
            dimensionXlim.push_back( val_2 );
//            if( val_3 != 7 ) {
//                dimensionXlim.push_back( val_3 );
//            }
        }
    }
    for( auto &el : dimensionXlim ) {
        double val = static_cast<double>( el );
        dimensionLong.push_back( val );
    }

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCsparse", dimensionLong }//,
    };
    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCsparse", {} }//,
    };
    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
#ifdef USECOLORGRAPHS
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } }//,
#else
        { "JVCsparse", { { "color", "black" }, {"marker", "o"}, { "linestyle", "--" }, { "linewidth", "1.5" }, { "label", "JVCsparse" } } }
#endif
    };

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPM::Sparse::CMatrixCOO mat_JVC_COO;
        SPM::Sparse::CMatrixCSR mat_JVC_CSR;

        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCsparse = 0.0;

        std::map<std::string, SPM::Timing::CTimeKeeper> timer = {
            { "JVCsparse", SPM::Timing::CTimeKeeper() }//,
        };

        double infValue = 1e7;
        double halfBigValue = -1e5;

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < ( 2 * n ); j++ ) {
                    if( j <= i ) {
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        if( randomDouble < levelCut ) { // ТУТ ВЕРНО! должно быть <
                            mat_JVC_COO.coo_val.push_back( randomDouble * scaleFactor );
                            mat_JVC_COO.coo_row.push_back( i );
                            mat_JVC_COO.coo_col.push_back( j );
                        }
                    } else if( ( j - n ) == i ) {
                        mat_JVC_COO.coo_val.push_back( halfBigValue );
                        mat_JVC_COO.coo_row.push_back( i );
                        mat_JVC_COO.coo_col.push_back( j );
                    }
                }
            }
            SPM::Sparse::MatrixCOOtoCSR( mat_JVC_COO, mat_JVC_CSR );

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPM::LAP::JVCsparse( mat_JVC_CSR.csr_val, mat_JVC_CSR.csr_kk, mat_JVC_CSR.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() ); // с
        }
    } // for n
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
//    for( auto &t : timeOfMethod ) {
//        plt::plot(
//            dimensionDouble.at( t.first ),
//            ( t.second ), // Y
//            estimated_keywords.at( t.first )
//            );
//    }

    for( auto &t : timeOfMethod ) {
        std::vector<double> dim = dimensionDouble.at( t.first );
        for( auto &ii : dim ) {
            ii *= 2;
        }
        plt::plot(
//            dimensionDouble.at( t.first ),
            dim,
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::xlim( ( dimensionXlim.front() * 2 ), dimensionXlim.back() );
    plt::ylim( -0.2, 2.0 );
    std::vector<int> dimensionXlimDiv;

    for( std::size_t i = 1; i < dimensionXlim.size(); i++ ) {
        if( ( i % 2 ) == 0 ) {
            dimensionXlimDiv.push_back( dimensionXlim[i] );
        }
    }
    plt::xticks( dimensionXlimDiv );
//    plt::xticks( dimensionXlim );
    plt::legend();
    plt::save( "sparseLargeExtra.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "sparseLargeExtra.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

//BOOST_AUTO_TEST_CASE( measureTime )
//{
//    SPM::Timing::CTimeKeeper tk;
//    tk.StartTimer();
//    for( int i = 0; i < 100; i++ ) {
//        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
//    }
//    tk.EndTimer();
//    std::cout << tk.TimePerOp() << std::endl;
//}

BOOST_AUTO_TEST_CASE( sparseLargeExtra2 )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;
    bool print = true; //false; //

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    std::uniform_real_distribution<double> random_double_probability( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

    std::vector<int> dimMarks = { 100, 500, 1000, 1500, 2000 };
    std::vector<int> dimTracks = { 3, 50, 100, 150, 200 };

    int cycle_count_ = 10;//100;//CYCLE_SHORT;//
    double scaleFactor = 1.0;//00000.0;
    double infValue = 1e7;
    double halfBigValue = -1e5;
    double lapcostJVCsparse = 0.0;
//    double levelCutting = 0.02;

    std::vector<double> levelCuttingVector = { 0.02, 0.3, 0.6, 0.98 };
//    std::vector<double> levelCuttingVector = { 0.02 };//, 0.3, 0.6, 0.98 };

    for( auto &levelCutting : levelCuttingVector ) {
        std::map<std::pair<int, int>, SPM::Timing::CTimeKeeper> timer;
        for( auto &m : dimMarks ) {
            for( auto &t : dimTracks ) {
                timer.insert( std::make_pair( std::make_pair( m, t ), SPM::Timing::CTimeKeeper() ) );
                for( int cycle = 0; cycle < cycle_count_; cycle++ ) {
                    if( print ) {
                        std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count_ << std::endl;
                    }
                    gen.seed( cycle );
                    // Заполним матрицу рандомными числами:
                    SPM::Sparse::CMatrixCOO mat_JVC_COO;
                    SPM::Sparse::CMatrixCSR mat_JVC_CSR;
                    for( int m_ = 0; m_ < m; m_++ ) {
                        for( int t_ = 0; t_ < ( t + m ); t_++ ) {
                            if( t_ <= m_ ) {
                                double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                                if( randomDouble < levelCutting ) { // ТУТ ВЕРНО! должно быть <
                                    mat_JVC_COO.coo_val.push_back( randomDouble * scaleFactor );
                                    mat_JVC_COO.coo_row.push_back( m_ );
                                    mat_JVC_COO.coo_col.push_back( t_ );
                                }
                            } else if( ( t_ - dimTracks.size() ) == m_ ) {
                                mat_JVC_COO.coo_val.push_back( halfBigValue );
                                mat_JVC_COO.coo_row.push_back( m_ );
                                mat_JVC_COO.coo_col.push_back( t_ );
                            }
                        }
                    }
                    SPM::Sparse::MatrixCOOtoCSR( mat_JVC_COO, mat_JVC_CSR );
                    arma::ivec actualJVCsparse = arma::ivec( m, arma::fill::zeros ); // Построчно!

                    timer.at( std::make_pair( m, t ) ).StartTimer();
                    int resSparse = SPM::LAP::JVCsparse( mat_JVC_CSR.csr_val, mat_JVC_CSR.csr_kk, mat_JVC_CSR.csr_first,
                        sp, infValue, resolution, actualJVCsparse, lapcostJVCsparse );
                    timer.at( std::make_pair( m, t ) ).EndTimer();
                    if( resSparse == 1 ) {
                        assert( false );
                    }
                } // for cycle
                if( print ) {
                    std::cout << " m = " << m << " t = " << t << " time = " << timer.at( std::make_pair( m, t ) ).TimePerOp() * 1e3 << "[ms]" << std::endl;
                }
            } // for t
        } // for m
        // Print matrix

        // Оптимизировано для вывода в таблицу для документации
        std::ofstream os;
        std::string fileName = "time_" + std::to_string( levelCutting ) + ".txt";
        os.open( fileName, std::ofstream::out );
        os << std::fixed << std::setprecision( 3 );

        bool printT = true;
        for( auto &m : dimMarks ) {
            if( printT ) {
                os << "отм./тр.";
                for( auto &t : dimTracks ) {
                    os << "\t" << t;
                }
                os << std::endl;
                printT = false;
            }
            os << m;
            for( auto &t : dimTracks ) {
                os << "\t" << timer.at( std::make_pair( m, t ) ).TimePerOp() * 1e3; // в мс
            }
            os << std::endl;
        }
        os.close();
    }

}

BOOST_AUTO_TEST_CASE( time_table )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;

    bool print = true; //false; //
    int cycle_count = CYCLE_SHORT;//1;//

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    std::uniform_real_distribution<double> random_double_probability( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;

    const double in2mm = 25.4;// mm (fixed)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, мс" );
#endif

    std::vector<double> dimensionLong;
    std::vector<double> dimensionShort;

    std::vector<int> dimensionXlim;
    int startPow = 2;
    int endPow = 10;
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
//        if( i < endPow ) {
//            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );

//            int val_1 = value + ( ( valueNext - value ) / 4 );
//            int val_2 = value + ( ( valueNext - value ) / 2 );
//            int val_3 = value + 3 * ( ( valueNext - value ) / 4 );
//            if( val_1 != 5 ) {
//                dimensionXlim.push_back( val_1 );
//            }
//            dimensionXlim.push_back( val_2 );
//            if( val_3 != 7 ) {
//                dimensionXlim.push_back( val_3 );
//            }
//        }
    }
    for( auto &el : dimensionXlim ) {
        double val = static_cast<double>( el );
        dimensionLong.push_back( val );
        if( val < 256 ) {
            dimensionShort.push_back( val );
        }
    }

    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLong },
        { "JVCsparse", dimensionLong },
        { "Mack", dimensionShort },
        { "Hungarian", dimensionShort },
        { "SeqExtr", dimensionShort },
        { "SeqExtrCOO", dimensionShort }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "JVCsparse", {} },
        { "Mack", {} },
        { "Hungarian", {} },
        { "SeqExtr", {} },
        { "SeqExtrCOO", {} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } },
        { "SeqExtr", { { "color", "orange" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtr" } } },
        { "SeqExtrCOO", { { "color", "yellow" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtrCOO" } } }
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPM::Sparse::CMatrixCSR mat_JVCsparse;
        SPM::Sparse::CMatrixCSR mat_JVCsparse2N;
        SPM::Sparse::CMatrixCOO mat_COOsparse;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_SeqExtr2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtr2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtrCOO2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCdense = 0.0, lapcostJVCsparse = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0,
            lapcostSeqExtr = 0.0, lapcostSeqExtrCOO = 0.0;

        std::map<std::string, SPM::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPM::Timing::CTimeKeeper() },
            { "JVCsparse", SPM::Timing::CTimeKeeper() },
            { "Mack", SPM::Timing::CTimeKeeper() },
            { "Hungarian", SPM::Timing::CTimeKeeper() },
            { "SeqExtr", SPM::Timing::CTimeKeeper() },
            { "SeqExtrCOO", SPM::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }
            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;

            double infValue = 1e7;
            double bigValue = -1e6;
            double halfBigValue = -1e5;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                double levelCut = 0.02; // Порог прореживания
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = bigValue;
                            mat_JVCdenseForSparse(i,j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPM::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::ones );
            mat_JVCdense2N *= bigValue;
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= bigValue;
            arma::vec emptyDenseDiag = arma::vec( n, arma::fill::ones );
            emptyDenseDiag *= halfBigValue;
            emptyDense.diag() = emptyDenseDiag;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
            emptySparse *= halfBigValue;

            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
            verbotenSparse *= halfBigValue;

            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT
//            mat_JVCdense2N.print( "mat_JVCdense2N" );
//            mat_JVCdenseForSparse2N.print( "mat_JVCdenseForSparse2N" );
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPM::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            mat_SeqExtr2N = mat_JVCdense2N;
            SPM::Sparse::MatrixDenseToCOO( mat_JVCdenseForSparse2N, mat_COOsparse );

            // Запуск решений с замерами времени выполнения
            timer.at( "JVCdense" ).StartTimer();
            SPM::LAP::JVCdense( mat_JVCdense2N, 2*n, sp, infValue, resolution, actualJVCdense2N, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPM::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            if( n <= boundN ) {
                timer.at( "Mack" ).StartTimer();
                SPM::LAP::Mack( mat_Mack2N, 2*n, sp, infValue, resolution, actualMack2N, lapcostMack );
                timer.at( "Mack" ).EndTimer();

                timer.at( "Hungarian" ).StartTimer();
                SPM::LAP::Hungarian( mat_Hungarian2N, 2*n, sp, infValue, resolution, actualHungarian2N, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();

                timer.at( "SeqExtr" ).StartTimer();
                SPM::LAP::SequentalExtremum( mat_SeqExtr2N, sp, infValue, resolution, actualSeqExtr2N, lapcostSeqExtr );
                timer.at( "SeqExtr" ).EndTimer();

                timer.at( "SeqExtrCOO" ).StartTimer();
                SPM::LAP::SequentalExtremum( mat_COOsparse, sp, infValue, resolution, actualSeqExtrCOO2N, lapcostSeqExtrCOO );
                timer.at( "SeqExtrCOO" ).EndTimer();
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
            std::vector< std::pair<int, int> > solutionMack;
            std::vector< std::pair<int, int> > solutionHungarian;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCdense2N(i) < n ) {
                    solutionJVCdense.push_back( std::make_pair( i, actualJVCdense2N(i) ) );
                }
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
                if( n <= boundN ) {
                    if( actualMack2N(i) < n ) {
                        solutionMack.push_back( std::make_pair( i, actualMack2N(i) ) );
                    }

                    if( actualHungarian2N(i) < n ) {
                        solutionHungarian.push_back( std::make_pair( i, actualHungarian2N(i) ) );
                    }
                }
            }

            if( ( solutionJVCdense.size() != solutionJVCsparse.size() ) ||
                ( ( n <= boundN ) && ( solutionJVCdense.size() != solutionMack.size() ) ) ||
                ( ( n <= boundN ) && ( solutionJVCdense.size() != solutionHungarian.size() ) ) )
            {
                counter_assign++;
            } else {
                int size = solutionJVCdense.size();
                for( int i = 0; i < size; i++ ) {
                    if( ( solutionJVCdense[i].first != solutionJVCsparse[i].first ) ||
                        ( solutionJVCdense[i].second != solutionJVCsparse[i].second ) ||
                        ( ( n <= boundN ) && ( ( solutionJVCdense[i].first != solutionMack[i].first ) ||
                        ( solutionJVCdense[i].second != solutionMack[i].second ) ) ) ||
                        ( ( n <= boundN ) && ( ( solutionJVCdense[i].first != solutionHungarian[i].first ) ||
                        ( solutionJVCdense[i].second != solutionHungarian[i].second ) ) ) )
                    {
                        counter_assign++;
                        break;
                    }
                }
            }

            // Оценим LAPCOST - суммарную стоимость назначения

            double totalcostJVCdense = 0.0;
            for( auto &elem : solutionJVCdense ) {
                totalcostJVCdense += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostJVCsparse = 0.0;
            for( auto &elem : solutionJVCsparse ) {
                totalcostJVCsparse += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostHungarian = 0.0;
            for( auto &elem : solutionHungarian ) {
                totalcostHungarian += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostMack = 0.0;
            for( auto &elem : solutionMack ) {
                totalcostMack+= mat_JVCdense( elem.first, elem.second );
            }

            bool eq_totalcostJVCdense_totalcostJVCsparse = std::abs( totalcostJVCdense - totalcostJVCsparse ) < 1e-5;
            bool eq_totalcostJVCdense_totalcostHungarian = false;
            bool eq_totalcostJVCdense_totalcostMack = false;
            if( n <= boundN ) {
                eq_totalcostJVCdense_totalcostHungarian = std::abs( totalcostJVCdense - totalcostHungarian ) < 1e-5;
                eq_totalcostJVCdense_totalcostMack = std::abs( totalcostJVCdense - totalcostMack ) < 1e-5;
            }

            if( !eq_totalcostJVCdense_totalcostJVCsparse ||
                ( ( n <= boundN ) && ( !eq_totalcostJVCdense_totalcostMack ) ) ||
                ( ( n <= boundN ) && ( !eq_totalcostJVCdense_totalcostHungarian ) ) )
            {
                counter_lapcost++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( ( n > boundN ) && ( t.first == "Hungarian" ) ) ||
                ( ( n > boundN ) && ( t.first == "SeqExtr" ) ) ||
                ( ( n > boundN ) && ( t.first == "SeqExtrCOO" ) ) ||
                ( ( n > boundN ) && ( t.first == "Mack" ) ) )
            {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
//    for( auto &t : timeOfMethod ) {
//        plt::plot(
//            dimensionDouble.at( t.first ),
//            ( t.second ), // Y
//            estimated_keywords.at( t.first )
//            );
//    }
    for( auto &t : timeOfMethod ) {
        std::vector<double> dim = dimensionDouble.at( t.first );
        for( auto &ii : dim ) {
            ii *= 2;
        }
        plt::plot(
//            dimensionDouble.at( t.first ),
            dim,
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::legend();
    plt::save( "time.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
//#ifdef PRINTTOTXT
    // Оптимизировано для вывода в таблицу для документации
    std::ofstream os3;
    os3.open( "time_table.txt", std::ofstream::out );
    os3 << "| N ";
    for( int i = 1; i < dimensionXlim.size(); i++ ) {
        os3 << "| " << dimensionXlim[i] << " ";
    }
    os3 << "|" << std::endl;

    for( int i = 1; i <= dimensionXlim.size(); i++ ) {
        os3 << "| :-: ";
    }
    os3 << "|" << std::endl;

    std::vector<std::string> methods3 = { "Hungarian", "Mack", "SeqExtr", "SeqExtrCOO", "JVCdense", "JVCsparse" };

    for( auto &m : methods3 ) {
        os3 << "| " << m << " ";
        for( int i = 0; i < dimensionXlim.size() - 1; i++ ) {
            for( auto &t : timeOfMethod ) {
                if( t.first == m ) {
                    if( i < ( ( t.second ).size() ) ) {
                        os3 << std::fixed << std::setprecision( 3 ) << "| " <<  ( ( t.second )[i] ) << " ";
                    } else {
                        os3 << "| - ";
                    }
                }
            }
        }
        os3 << "|" << std::endl;
    }
    os3.close();

    // Оптимизировано для вывода в таблицу для документации
    std::ofstream os4;
    os4.open( "time_table_relative.txt", std::ofstream::out );
    os4 << "| N ";
    for( int i = 1; i < dimensionXlim.size(); i++ ) {
        os4 << "| " << dimensionXlim[i] << " ";
    }
    os4 << "|" << std::endl;

    for( int i = 1; i <= dimensionXlim.size(); i++ ) {
        os4 << "| :-: ";
    }
    os4 << "|" << std::endl;

    std::vector<std::string> methods4 = { "Hungarian", "Mack", "SeqExtr", "SeqExtrCOO", "JVCdense" };

    for( auto &m : methods4 ) {
        os4 << "| " << m << " ";
        for( int i = 0; i < dimensionXlim.size() - 1; i++ ) {
            for( auto &t : timeOfMethod ) {
                if( t.first == m ) {
                    if( i < ( ( t.second ).size() ) ) {
                        os4 << std::fixed << std::setprecision( 3 ) << "| " <<  ( ( t.second )[i] ) / ( timeOfMethod.at( "JVCsparse" ))[i] << " ";
                    } else {
                        os4 << "| - ";
                    }
                }
            }
        }
        os4 << "|" << std::endl;
    }
    os4.close();
//#endif
}

