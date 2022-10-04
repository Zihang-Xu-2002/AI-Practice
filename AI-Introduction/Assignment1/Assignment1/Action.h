#pragma once
#include <iostream>
//这里定义了“上下左右、失败、减去”六种行为
enum class Action
{
	LEFT,
	UP,
	RIGHT,
	DOWN,
	FAILURE,
	CUTOFF
};

extern std::ostream& operator<<(std::ostream& os, Action action);