//----------------------------------------------------------------------------------------------------------------------
///
/// \file       jvc_dense.hpp
/// \brief      Решение задачи о назначениях методом JVC для плотных матриц
/// \date       07.02.23 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#ifndef SPML_JVC_DENSE_HPP_
#define SPML_JVC_DENSE_HPP_

#include <armadillo>

#include <compare.hpp>
#include <searchparam.hpp>

namespace SPML /// Специальная библиотека программных модулей (СБПМ)
{
namespace LAP /// Решение задачи о назначениях
{
///
/// \brief Метод Джонкера-Волгенанта-Кастаньона (Jonker-Volgenant-Castanon) решения задачи о назначениях для
/// плотных матриц
/// \details Источники:
/// 1)  "A Shortest Augmenting Path Algorithm for Dense and Sparse Linear
///     Assignment Problems," Computing 38, 325-340, 1987 by
///     R. Jonker and A. Volgenant, University of Amsterdam.
/// 2)  https://github.com/yongyanghz/LAPJV-algorithm-c
/// 3)  https://www.mathworks.com/matlabcentral/fileexchange/26836-lapjv-jonker-volgenant-algorithm-for-linear-assignment-problem-v3-0
/// \remarks - Оригинальный код R. Jonker and A. Volgenant [1] для целых чисел адаптирован под вещественные
///          - Метод подразделен на 4 процедуры в соответствии с модификацией Castanon:
///            * COLUMN REDUCTION
///            * REDUCTION TRANSFER
///            * AUGMENTING ROW REDUCTION - аукцион
///            * AUGMENT SOLUTION FOR EACH FREE ROW на основе алгоритма Dijkstra
///          - Правки из [3] касающиеся точности сравнения вещественных чисел
/// \param[in] assigncost - квадратная матрица ценности, размер [dim,dim]
/// \param[in] dim - порядок квадратной матрицы ценности и размерность результата res соответственно
/// \param[in] sp - критерий поиска (минимум/максимум)
/// \param[in] infValue - большое положительное число
/// \param[in] resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// "rowsol[i] = j" означает, что в i-ой строке выбран j-ый элемент
/// \param[out] lapcost - сумма назначенных элементов матрицы ценности assigncost
///
void JVCdense( const arma::mat &assigncost, int dim, TSearchParam sp, double infValue, double resolution,
    arma::ivec &rowsol, double &lapcost );

} // namespace LAP
} // namespace SPML
#endif
/// \}
