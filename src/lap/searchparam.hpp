//----------------------------------------------------------------------------------------------------------------------
///
/// \file       searchparam.hpp
/// \brief      Параметр поиска max/min
/// \details    Вынесено отдельно от реализации методов решения задачи о назначениях
/// \date       07.02.23 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#ifndef SPML_SEARCHPARAM_HPP_
#define SPML_SEARCHPARAM_HPP_

namespace SPML /// Специальная библиотека программных модулей (СБПМ)
{
namespace LAP /// Решение задачи о назначениях
{
///
/// \brief Критерий поиска - минимум/максимум для задачи о назначениях
///
enum TSearchParam
{
    SP_Min,
    SP_Max
};

} // namespace LAP
} // namespace SPML
#endif
/// \}
