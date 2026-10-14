//----------------------------------------------------------------------------------------------------------------------
///
/// \file       murty.cpp
/// \brief      Решение k-best задачи о назначениях методом Murty
/// \details    Единичная задача о назначениях решается методом JVC для разреженных матриц
/// \date       21.01.26 - создан     
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#include <murty.hpp>

#include <algorithm>
#include <set>

namespace SPML 
{
namespace LAP 
{
namespace {

bool isBanned( const std::vector<std::pair<int,int>>& banned, int row, int col )
{
    for( const auto& edge : banned ) {
        if( edge.first == row && edge.second == col ) {
            return true;
        }
    }
    return false;
}

bool buildConstrainedCsr(
    const Sparse::CMatrixCSR& src,
    const std::vector<int>& fixedCol,
    const std::vector<std::pair<int,int>>& banned,
    Sparse::CMatrixCSR& dst )
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

static bool MurtyCmpMax( const MurtyNode &a, const MurtyNode &b ) {
    return a.lb < b.lb; // для поиска Max
}

static bool MurtyCmpMin( const MurtyNode &a, const MurtyNode &b ) {
    return a.lb > b.lb; // для поиска Min
}
  
Murty::Murty( const Sparse::CMatrixCSR& csr, TSearchParam sp, double inf, double res )
    : csr_( csr )
    , sp_( sp )
    , inf_( inf )
    , res_( res )
    , initialized_( false )
    , pq_( ( sp == TSearchParam::SP_Max ) ? MurtyCmpMax : MurtyCmpMin )
{}

MurtyNode Murty::solveNode( const MurtyNode* parent, int split_row )
{
    MurtyNode n;
    n.split_from = std::max( split_row, 0 );
    const int nr = csr_.n_rows();

    n.fixed_col.assign( nr, -1 );

    if( parent ) {
        // OLD
        // n.fixed_col = parent->fixed_col;

        // NEW
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
    if( !buildConstrainedCsr( csr_, n.fixed_col, n.banned, constrained ) ) {
        n.sol.cost = inf_;
        return n;
    }

    arma::ivec x( nr );
    arma::vec u_out( nr ), v_out( constrained.n_cols() );
    double cost = 0.0;

    int rc = Murty_JVCsparse( constrained, sp_, inf_, res_, x, cost, nullptr, nullptr, &u_out, &v_out );

    if( rc != 0 || cost >= inf_ ) {
        n.sol.cost = inf_;
        return n;
    }

    n.sol.x = x;
    n.sol.u = std::move( u_out );
    n.sol.v = std::move( v_out );
    n.sol.cost = cost;
    n.lb = cost;
    return n;
}

bool Murty::findNext( MurtySolution& out )
{
    if( !initialized_ ) {
        MurtyNode root = solveNode( nullptr, -1 );
        if( root.sol.cost >= inf_ ) {
            return false;
        }
        pq_.push( root );
        initialized_ = true;
    }

    if (pq_.empty()) return false;

    MurtyNode best = pq_.top(); // Лучший - на первой позиции!
    pq_.pop();
    out = best.sol;

    const int nr = best.sol.x.n_elem;
    // for (int i = 0; i < nr; ++i) {
    for( int i = best.split_from; i < nr; ++i ) {
        MurtyNode child = solveNode( &best, i );
        if( child.sol.cost < inf_ ) {
            pq_.push( child );
        }
    }
    return true;
}

} // end namespace LAP
} // end namespace SPML
/// \}
