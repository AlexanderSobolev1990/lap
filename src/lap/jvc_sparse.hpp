//----------------------------------------------------------------------------------------------------------------------
///
/// \file       jvc_sparse.hpp
/// \brief      Решение задачи о назначениях методом JVC для разреженных матриц
/// \date       07.02.23 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#ifndef SPML_JVC_SPARSE_HPP_
#define SPML_JVC_SPARSE_HPP_

#include <armadillo>

#include <compare.hpp>
#include <searchparam.hpp>
#include <sparse.hpp>

namespace SPML /// Специальная библиотека программных модулей (СБПМ)
{
namespace LAP /// Решение задачи о назначениях
{
///
/// \brief Метод Джонкера-Волгенанта-Кастаньона (Jonker-Volgenant-Castanon) решения задачи о назначениях для
/// разреженных матриц
/// \param[in] cc - вектор ненулевых элементов матрицы
/// \param[in] kk - вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов
/// \param[in] first - вектор начальных смещений в векторе сс
/// \param[in] sp - критерий поиска (минимум/максимум)
/// \param[in] infValue - большое положительное число
/// \param[in] resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// "rowsol[i] = j" означает, что в i-ой строке выбран j-ый элемент
/// \param[out] lapcost - сумма назначенных элементов матрицы ценности assigncost
/// \return 0 - OK, 1 - fail
///
int JVCsparse( const std::vector<double> &cc, const std::vector<int> &kk, const std::vector<int> &first,
    TSearchParam sp, double infValue, double resolution, arma::ivec &rowsol, double &lapcost );

///
/// \brief Метод Джонкера-Волгенанта-Кастаньона (Jonker-Volgenant-Castanon) решения задачи о назначениях для
/// разреженных матриц
/// \param[in] csr - матрица в CSR формате (Compressed Sparse Row Yale format)
/// \param[in] sp - критерий поиска (минимум/максимум)
/// \param[in] infValue - большое положительное число
/// \param[in] resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// "rowsol[i] = j" означает, что в i-ой строке выбран j-ый элемент
/// \param[out] lapcost - сумма назначенных элементов матрицы ценности assigncost
/// \return 0 - OK, 1 - fail
///
int JVCsparse( const Sparse::CMatrixCSR &csr, TSearchParam sp, double infValue, double resolution, arma::ivec &rowsol,
    double &lapcost );

} // namespace LAP
} // namespace SPML
#endif
/// \}
