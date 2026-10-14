//----------------------------------------------------------------------------------------------------------------------
///
/// \file       test_murty.cpp
/// \brief      Regression tests for Murty K-best assignment.
///

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_murty

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <vector>

#include <armadillo>
#include <boost/test/unit_test.hpp>

#include <murty.hpp>
#include <sparse.hpp>

namespace {

struct BruteSolution
{
    arma::ivec x;
    double cost = 0.0;
};

double denseCost( const arma::mat& cost, const arma::ivec& x )
{
    double result = 0.0;
    for( arma::uword i = 0; i < x.n_elem; ++i ) {
        result += cost( i, x( i ) );
    }
    return result;
}

std::string keyOf( const arma::ivec& x )
{
    std::ostringstream ss;
    for( arma::uword i = 0; i < x.n_elem; ++i ) {
        if( i > 0 ) {
            ss << ',';
        }
        ss << x( i );
    }
    return ss.str();
}

void enumerateAssignments(
    const arma::mat& cost,
    const double empty,
    const int row,
    std::vector<int>& used,
    arma::ivec& current,
    std::vector<BruteSolution>& out )
{
    if( row == static_cast<int>( cost.n_rows ) ) {
        BruteSolution sol;
        sol.x = current;
        sol.cost = denseCost( cost, current );
        out.push_back( sol );
        return;
    }

    for( arma::uword col = 0; col < cost.n_cols; ++col ) {
        if( used[col] != 0 ) {
            continue;
        }
        if( std::abs( cost( row, col ) - empty ) < 1e-12 ) {
            continue;
        }
        used[col] = 1;
        current( row ) = static_cast<int>( col );
        enumerateAssignments( cost, empty, row + 1, used, current, out );
        used[col] = 0;
    }
}

std::vector<BruteSolution> bruteForce(
    const arma::mat& cost,
    const double empty,
    const SPML::LAP::TSearchParam searchParam )
{
    std::vector<BruteSolution> out;
    std::vector<int> used( cost.n_cols, 0 );
    arma::ivec current( cost.n_rows, arma::fill::zeros );
    enumerateAssignments( cost, empty, 0, used, current, out );

    std::sort( out.begin(), out.end(),
        [searchParam]( const BruteSolution& lhs, const BruteSolution& rhs ) {
            if( std::abs( lhs.cost - rhs.cost ) > 1e-9 ) {
                return ( searchParam == SPML::LAP::TSearchParam::SP_Max )
                    ? ( lhs.cost > rhs.cost )
                    : ( lhs.cost < rhs.cost );
            }
            return keyOf( lhs.x ) < keyOf( rhs.x );
        } );
    return out;
}

void checkMurtyAgainstBruteForce(
    const arma::mat& cost,
    const double empty,
    const SPML::LAP::TSearchParam searchParam,
    const int kBest )
{
    SPML::Sparse::CMatrixCSR csr;
    SPML::Sparse::MatrixDenseToCSR( cost, csr, empty, 1e-12 );

    const double infValue = 1e12;
    const double resolution = 1e-9;
    SPML::LAP::Murty murty( csr, searchParam, infValue, resolution );
    const std::vector<BruteSolution> expected = bruteForce( cost, empty, searchParam );
    std::set<std::string> seen;

    BOOST_REQUIRE_GE( static_cast<int>( expected.size() ), kBest );
    for( int k = 0; k < kBest; ++k ) {
        SPML::LAP::MurtySolution actual;
        BOOST_REQUIRE( murty.findNext( actual ) );
        BOOST_CHECK_SMALL( actual.cost - expected[k].cost, 1e-7 );
        BOOST_CHECK_EQUAL( seen.insert( keyOf( actual.x ) ).second, true );
    }
}

} // namespace

BOOST_AUTO_TEST_CASE( murty_dense_min_matches_bruteforce )
{
    const arma::mat cost = {
        { 9.1, 2.2, 7.3, 8.4 },
        { 6.5, 4.6, 3.7, 7.8 },
        { 5.9, 8.0, 1.1, 8.2 },
        { 7.3, 6.4, 9.5, 4.6 }
    };

    checkMurtyAgainstBruteForce( cost, 0.0, SPML::LAP::TSearchParam::SP_Min, 12 );
}

BOOST_AUTO_TEST_CASE( murty_dense_max_matches_bruteforce )
{
    const arma::mat cost = {
        { 0.61, 0.32, 0.13, 0.24, 0.45 },
        { 0.92, 0.56, 0.79, 0.66, 0.15 },
        { 0.33, 0.88, 0.43, 0.11, 0.58 },
        { 0.77, 0.19, 0.54, 0.82, 0.22 }
    };

    checkMurtyAgainstBruteForce( cost, 0.0, SPML::LAP::TSearchParam::SP_Max, 20 );
}

BOOST_AUTO_TEST_CASE( murty_sparse_min_matches_bruteforce )
{
    const double empty = 1e9;
    arma::mat cost = {
        { 4.1, empty, 7.3, 2.4, empty },
        { empty, 3.6, 5.7, empty, 9.8 },
        { 6.9, 8.0, empty, 1.2, 4.3 },
        { 7.4, empty, 2.5, 6.6, 3.7 }
    };

    checkMurtyAgainstBruteForce( cost, empty, SPML::LAP::TSearchParam::SP_Min, 10 );
}
