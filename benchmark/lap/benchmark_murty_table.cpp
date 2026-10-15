//----------------------------------------------------------------------------------------------------------------------
///
/// \file       benchmark_murty_table.cpp
/// \brief      Получение таблицы решений Murty для примера 3x6 из README
///

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "spm/lap/murty.hpp"
#include "spm/lap/sparse.hpp"

int main() try
{
    const arma::mat cost = {
        { 0.6, 0.3, 0.0, 0.1, 0.0, 0.0 },
        { 0.0, 0.2, 0.7, 0.0, 0.1, 0.0 },
        { 0.3, 0.5, 0.1, 0.0, 0.0, 0.1 }
    };
    SPM::Sparse::CMatrixCSR csr;
    // В этом примере ноль означает отсутствующее ребро.
    SPM::Sparse::MatrixDenseToCSR( cost, csr, 0.0, 1e-12 );
    const auto sp = SPM::LAP::SP_Max;
    const double infValue = 1e7;
    const double resolution = 1e-7;
    std::vector<SPM::LAP::MurtySolution> solutions;
    SPM::LAP::Murty reference( csr, sp, infValue, resolution );
    SPM::LAP::MurtySolution solution;
    while( reference.findNext( solution ) ) {
        solutions.push_back( solution );
        if( solutions.size() > 22 ) {
            throw std::runtime_error( "Unexpected number of assignments for the README example" );
        }
    }
    if( solutions.size() != 22 ) {
        throw std::runtime_error( "Expected all 22 assignments for the README example" );
    }

    const int warmupCount = 100;
    const int sampleCount = 1001;
    std::vector<std::vector<double>> samples( solutions.size() );
    for( auto &values : samples ) {
        values.reserve( sampleCount );
    }
    for( int run = -warmupCount; run < sampleCount; ++run ) {
        SPM::LAP::Murty murty( csr, sp, infValue, resolution );
        for( std::size_t rank = 0; rank < solutions.size(); ++rank ) {
            const auto start = std::chrono::steady_clock::now();
            const bool found = murty.findNext( solution );
            const auto finish = std::chrono::steady_clock::now();
            if( !found || std::abs( solution.cost - solutions[rank].cost ) > 1e-12
                || !arma::all( solution.x == solutions[rank].x ) ) {
                throw std::runtime_error( "Assignment sequence changed between benchmark runs" );
            }
            if( run >= 0 ) {
                samples[rank].push_back( std::chrono::duration<double, std::micro>( finish - start ).count() );
            }
        }
        if( murty.findNext( solution ) ) {
            throw std::runtime_error( "Assignment enumeration did not finish after 22 solutions" );
        }
    }

    std::cout << "| K-best | Cost | Rowsol | Time, us (median) |\n"
        << "| :-: | :-: | :-: | :-: |\n";
    for( std::size_t rank = 0; rank < solutions.size(); ++rank ) {
        auto &values = samples[rank];
        std::sort( values.begin(), values.end() );
        std::cout << "| " << rank + 1 << " | " << std::fixed << std::setprecision( 1 )
            << solutions[rank].cost << " |";
        for( arma::sword col : solutions[rank].x ) {
            std::cout << ' ' << col;
        }
        std::cout << " | " << std::setprecision( 3 ) << values[values.size() / 2] << " |\n";
    }
    return 0;
}
catch( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
