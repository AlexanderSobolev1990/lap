//----------------------------------------------------------------------------------------------------------------------
///
/// \file       lap_mack.cpp
/// \brief      Решение задачи о назначениях методом Мака (cтандартная линейная дискретная оптимизационная задача)
/// \date       18.05.22 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#include <lap.h>

namespace SPML /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace LAP /// Решение задачи о назначениях
{
//----------------------------------------------------------------------------------------------------------------------
void Mack( const arma::mat &assigncost, int dim, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol,
    double &lapcost )
{
    double* cost = new double[dim*dim];
    for( int i = 0; i < dim; i++ ) {
        for( int j = 0; j < dim; j++ ) {
            if( sp == TSearchParam::SP_Max ) {
                cost[dim*i+j] = -assigncost(i,j); // Поиск максимума
            } else {
                cost[dim*i+j] = assigncost(i,j); // Поиск минимума
            }
        }
    }

    double* ma = new double[dim+1];
    double* mb = new double[dim+1];
    int* ip = new int[dim+1];
    int* im = new int[dim+1];
    int* ic = new int[(dim+1) * (dim+1)];
    int* lr = new int[dim+1];

    int* jr = new int[dim+1];
    int* jm = new int[dim+1];
    int* jk = new int[dim+1];
    int* jv = new int[dim+1]; // result of algorithm (needs to be converted from 1...N fortran format to 0...(N-1) C# format)
    int* nm = new int[dim+1];

    // clear memory
    memset( ma, 0, static_cast<unsigned long long>( dim + 1 ) * sizeof(double) );
    memset( mb, 0, static_cast<unsigned long long>( dim + 1 ) * sizeof(double) );
    memset( ip, 0, static_cast<unsigned long long>( dim + 1 ) * sizeof(int) );
    memset( im, 0, static_cast<unsigned long long>( dim + 1 ) * sizeof(int) );
    memset( ic, 0, static_cast<unsigned long long>( ( dim + 1 ) * ( dim + 1 ) ) * sizeof(int) );
    memset( lr, 0, static_cast<unsigned long long>( dim + 1 ) * sizeof(int) );
    memset( jr, 0, static_cast<unsigned long long>( dim + 1 ) * sizeof(int) );
    memset( jm, 0, static_cast<unsigned long long>( dim + 1 ) * sizeof(int) );
    memset( jk, 0, static_cast<unsigned long long>( dim + 1 ) * sizeof(int) );
    memset( jv, 0, static_cast<unsigned long long>( dim + 1 ) * sizeof(int) );
    memset( nm, 0, static_cast<unsigned long long>( dim + 1 ) * sizeof(int) );

    double rim = 0;
    double riz = 0;
    double riv = 0;

    int nc = 0;

    int i = 0;
    int il = 0;
    int ir = 0;
    int iw = 0;
    int iz = 0;
    int icj = 0;
    int ilr = 0;
    int iip = 0;

    int l = 0;
    int ls = 0;
    int k = 0;

    int j = 0;
    int jc = 0;
    int jd = 0;
    int jp = 0;
    int jq = 0;
    int ju1 = 0;
    int jx = 0;
    int jy = 0;

    double rma = 1e10;

    // 2
    for( i = 1; i <= dim; i++ ) {
        rim = rma;
        for( j = 1; j <= dim; j++ ) {
            riz = cost[dim*(i-1)+(j-1)]; //riz = p[i, j];
            if( riz > rim ) {
                continue;
            }
            rim = riz;
            l = j;
        }
            nm[l] = nm[l] + 1;
            k = nm[l];
            ic[dim*(l-1)+(k-1)] = i; //ic[l, k] = i;
            ma[i] = rim;
            jr[i] = l;
    }
    bool isJN = false;
    for( ; ; ) {
        j = 0;
        for( ; ; ) {
            j = j + 1;
            if( j > dim ) {
                isJN = true;
                break;
            }
            if( nm[j] >= 2 ) {
                break;
            }
        }
        if( isJN ) {
            break;
        }
        ju1 = nm[j];

        for( i = 1; i <= dim; i++ ) {
            ip[i] = ic[dim*(j-1)+(i-1)]; //ip[i] = ic[j, i];
        }
        nc = 1;
        lr[1] = j;
        jk[j] = 1;
        mb[j] = 0;
        for( ; ; ) {
            riv = rma;
            // 4
            for( k = 1; k <= ju1; k++ ) {
                i = ip[k];
                for( jd = 1; jd <= dim; jd++ ) {
                    if( jk[jd] == 1 ) {
                        continue;
                    }
                    riz = cost[dim*(i-1)+(jd-1)] - ma[i]; //riz = p[i, jd] - ma[i];
                    if( riz > riv ) {
                        continue;
                    }
                    riv = riz;
                    jc = jd;
                    ir = i;
                }
            }
            // 5
            for( jx = 1; jx <= nc; jx++ ) {
                ilr = lr[jx];
                mb[ilr] = mb[ilr] + riv;
            }
            for( k = 1; k <= ju1; k++ ) {
                iip = ip[k];
                ma[iip] = ma[iip] + riv;
            }
            mb[jc] = 0;
            jk[jc] = 1;
            nc = nc + 1;
            lr[nc] = jc;
            im[jc] = ir;
            jm[ir] = jc;
            jy = nm[jc];
            if( jy != 0 ) {
                for( jx = 1; jx <= jy; jx++ ) {
                    ju1 = ju1 + 1;
                    ip[ju1] = ic[dim*(jc-1)+(jx-1)]; // ip[ju1] = ic[jc, jx];
                }
                continue;
            }
            break;
        }
        for( jx = 1; jx <= nc; jx++ ) {
            ls = lr[jx];
            jk[ls] = 0;
            for( i = 1; i <= dim; i++ ) {
                cost[dim*(i-1)+(ls-1)] = cost[dim*(i-1)+(ls-1)] + mb[ls]; //p[i, ls] = p[i, ls] + mb[ls];
            }
        }
        nm[jc] = 1;
        ic[dim*(jc-1)+1-1] = ir; // ic[jc, 1] = ir;
        for( ; ; ) {
            jp = jr[ir];
            jr[ir] = jc;
            iw = 0;
            jq = nm[jp];
            for( il = 1; il <= jq; il++ ) {
                iz = ic[dim*(jp-1)+(il-1)]; // iz = ic[jp, il];
                if( iz != ir ) {
                    iw = iw + 1;
                    ic[dim*(jp-1)+(iw-1)] = ic[dim*(jp-1)+(il-1)];// ic[jp, iw] = ic[jp, il];
                }
            }
            if( jq > 1 ) {
                break;
            }
            ir = im[jp];
            jc = jp;
            ic[dim*(jp-1)+(jq-1)] = ir;//ic[jp, jq] = ir;
        }
        nm[jp] = jq - 1;
    }
    for( j = 1; j <= dim; j++ ) {
        icj = ic[dim*(j-1)+1-1];// icj = ic[j, 1];
        jv[icj] = j;
    }

    // Вывод результата
    j = 0;
    lapcost = 0.0;
    for( int i = 0; i < dim; i++ ) {
        j = jv[i+1]-1;
        rowsol[i] = j; // в i-ой строке j-ый элемент
        double element_i_j = assigncost(i,j);
        if( !SPML::Compare::AreEqualAbs( element_i_j, maxcost, resolution ) ) {
            lapcost += element_i_j;
        }
    }

    // освобождаем память
    delete[] cost;
    delete[] ma;
    delete[] mb;
    delete[] ip;
    delete[] im;
    delete[] ic;
    delete[] lr;
    delete[] jr;
    delete[] jm;
    delete[] jk;
    delete[] jv;
    delete[] nm;
    return;
}

} // end namespace LAP
} // end namespace SPML
/// \}

