#include <iostream>


/**
 * \brief Math function of atan
 * \param[in] x Input of atan
 * \return Result of atan(0.8*x + 0.2)
 */
double returnAtan(const double x)
{
	constexpr double inaccPhi = 10e-6 / 5.4;

	double phiSummand = 0.8 * x + 0.2;
	double phi = phiSummand;

	int count = 1;
	do
	{
		phiSummand = phiSummand * -1 * (0.8 * x + 0.2) * (0.8 * x + 0.2)
		             * (2 * (count - 1) + 1) / (2 * count + 1);
		phi += phiSummand;

		++count;
	}
	while (std::abs(phiSummand) >= inaccPhi);

	return phi;
}

/**
* \brief Math function of exponent
* \param[in] x Input of exponent
* \return Result of e^(2*x + 1)
*/
double returnExp(const double x)
{
	constexpr double inaccG = 10e-6 / 3.51;

	double gSummand = 1;
	double g = gSummand;

	int count = 1;
	do
	{
		gSummand = gSummand * (2 * x + 1) / count;
		g += gSummand;

		++count;
	}
	while (std::abs(gSummand) >= inaccG);

	return g;
}


/**
 * \brief My realization of the function
 * \param x Input parameter of function, less than 1
 * \return Result of (1+atan(0.8*x + 0.2))^0.5 * e^(2*x + 1)
 */
double functionR(const double x)
{
	const double phi = returnAtan(x);
	const double g = returnExp(x);

	constexpr double inaccFunc = 10e-6 / 3;

	double relativeValPrev;
	double relativeVal = 0;
	double sqrtSummand = 3; // square root of function with excess

	do
	{
		relativeValPrev = relativeVal;
		sqrtSummand = 0.5 * (sqrtSummand + (1 + phi) / sqrtSummand);
		relativeVal = sqrtSummand * g;
	}
	while (std::abs(relativeVal - relativeValPrev) >= inaccFunc);

	return relativeVal;
}

/**
* \brief Realization of the function with built-in functions
* \param x Input parameter of function
* \return Result of (1+atan(0.8*x + 0.2))^0.5 * e^(2*x + 1)
*/
double functionComp(const double x)
{
	constexpr double e = 2.7182818284590452;
	return std::pow(1 + std::atan(0.8 * x + 0.2), 0.5) * std::pow(e, 2 * x + 1);
}


int main()
{
	std::cout << "x\tf(x)\tfa(x)\tdeltaf\n";

	for (double x = 0.1; x < 0.21; x += 0.01)
	{
		const double relativeVal = functionR(x);
		const double accurateVal = functionComp(x);
		

		std::cout << x << '\t' << accurateVal << '\t' << relativeVal << '\t'
				<< std::abs(accurateVal - relativeVal) << '\n';
	}

	std::cin.get();
	return 0;
}
