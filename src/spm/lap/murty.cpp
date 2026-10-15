//----------------------------------------------------------------------------------------------------------------------
///
/// \file       murty.cpp
/// \brief      Решение k-best задачи о назначениях методом Murty
/// \details    Единичная задача о назначениях решается методом JVC для разреженных матриц
/// \addtogroup spm
/// \{
///

#include "spm/lap/murty.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace SPM
{
namespace LAP
{
namespace {

bool isBanned( const std::vector<std::pair<int,int>> & banned, int row, int col )
{
    for( const auto & edge : banned ) {
        if( edge.first == row && edge.second == col ) {
            return true;
        }
    }
    return false;
}

bool buildConstrainedCsr(
    const Sparse::CMatrixCSR & src,
    const std::vector<int> & fixedCol,
    const std::vector<std::pair<int,int>> & banned,
    Sparse::CMatrixCSR & dst )
{
    const int nr = src.n_rows();
    std::set<int> fixedColumns;
    for( int row = 0; row < nr; ++row ) {
        const int col = fixedCol[row];
        if( col < 0 ) {
            continue;
        }
        if( isBanned( banned, row, col ) || fixedColumns.count( col ) != 0 ) {
            return false;
        }
        fixedColumns.insert( col );
    }

    dst.csr_val.clear();
    dst.csr_kk.clear();
    dst.csr_first.clear();
    dst.csr_first.reserve( src.csr_first.size() );
    dst.csr_val.reserve( src.csr_val.size() );
    dst.csr_kk.reserve( src.csr_kk.size() );
    dst.csr_first.push_back( 0 );

    for( int row = 0; row < nr; ++row ) {
        const int fixed = fixedCol[row];
        int added = 0;
        for( int t = src.csr_first[row]; t < src.csr_first[row + 1]; ++t ) {
            const int col = src.csr_kk[t];
            if( fixed >= 0 && col != fixed ) {
                continue;
            }
            if( fixed < 0 && fixedColumns.count( col ) != 0 ) {
                continue;
            }
            if( isBanned( banned, row, col ) ) {
                continue;
            }
            dst.csr_val.push_back( src.csr_val[t] );
            dst.csr_kk.push_back( col );
            ++added;
        }
        if( added == 0 ) {
            return false;
        }
        dst.csr_first.push_back( static_cast<int>( dst.csr_val.size() ) );
    }
    return true;
}

} // namespace

bool Murty::CompareMax( const MurtyNode &a, const MurtyNode &b )
{
    return a.sol.cost < b.sol.cost; // для поиска Max
}

bool Murty::CompareMin( const MurtyNode &a, const MurtyNode &b )
{
    return a.sol.cost > b.sol.cost; // для поиска Min
}

Murty::Murty( const Sparse::CMatrixCSR & csr, TSearchParam sp, double inf, double res )
    : csr_( csr )
    , sp_( sp )
    , inf_( inf )
    , res_( res )
    , initialized_( false )
    , pq_( ( sp == TSearchParam::SP_Max ) ? CompareMax : CompareMin )
{}

Murty::MurtyNode Murty::solveNode( const MurtyNode * parent, int split_row )
{
    MurtyNode n;
    n.split_from = std::max( split_row, 0 );
    const int nr = csr_.n_rows();
    if( nr < 0 ) {
        throw std::length_error( "LAP::Murty: csr_first must not be empty" );
    }

    n.fixed_col.assign( nr, -1 );

    if( parent ) {
        // Фиксируем все строки < split_row
        for( int i = 0; i < split_row; ++i ) {
            n.fixed_col[i] = parent->sol.x(i);
        }

        // Копируем предыдущие bans
        n.banned = parent->banned;

        // Запрещаем текущее ребро
        int c = parent->sol.x( split_row );
        n.banned.emplace_back( split_row, c );
    }

    Sparse::CMatrixCSR constrained;
    const Sparse::CMatrixCSR *input = &csr_;
    if( parent ) {
        if( !buildConstrainedCsr( csr_, n.fixed_col, n.banned, constrained ) ) {
            return n;
        }
        input = &constrained;
    }

    arma::ivec x( nr );
    arma::vec u_out, v_out;
    if( parent ) {
        u_out = parent->sol.u;
        v_out = parent->sol.v;
    }
    double cost = 0.0;

    int rc = JVCsparse( *input, sp_, inf_, res_, x, cost, &u_out, &v_out,
        parent ? &parent->sol.x : nullptr );

    if( rc != 0 ) {
        return n;
    }

    n.sol.x = std::move( x );
    n.sol.u = std::move( u_out );
    n.sol.v = std::move( v_out );
    n.sol.cost = cost;
    n.feasible = true;
    return n;
}

bool Murty::findNext( MurtySolution & out )
{
    if( !initialized_ ) {
        MurtyNode root = solveNode( nullptr, -1 );
        initialized_ = true;
        if( !root.feasible ) {
            return false;
        }
        pq_.push( std::move( root ) );
    }

    if( pq_.empty() ) {
        return false;
    }

    MurtyNode best = pq_.top(); // Лучший - на первой позиции!
    pq_.pop();
    out = best.sol;

    const int nr = best.sol.x.n_elem;
    for( int i = best.split_from; i < nr; ++i ) {
        MurtyNode child = solveNode( &best, i );
        if( child.feasible ) {
            pq_.push( std::move( child ) );
        }
    }
    return true;
}

} // namespace LAP
} // namespace SPM
/// \}
