#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "netpbm.h"


// Linearly convolve the intensity information of an image with a convolution filter (both 
// stored in Matrices). The center of the filter serves as its anchor for storing the result. 
// For even number of rows or columns, the anchor will be rounded to the left or upward, resp. 
// The output is a matrix of the same size as the original image, with zeroes filling the 
// border regions that are unreachable by the filter.
Matrix convolve(Matrix img, Matrix filter)
{
	int m, n, k, l, mOffset = filter.height / 2, nOffset = filter.width / 2;
	double sum;
	Matrix result = createMatrix(img.height, img.width);

	for (m = filter.height - 1; m < img.height; m++)
		for (n = filter.width - 1; n < img.width; n++)
		{
			sum = 0.0;
			for (k = 0; k < filter.width; k++)
				for (l = 0; l < filter.height; l++)
					sum += img.map[m - k][n - l] * filter.map[k][l];
			result.map[m - mOffset][n - nOffset] = sum;
		}
	return result;
}


Image sobel(Image img)
{
	double hFilter[3][3] = { { 1.0, 0.0, -1.0 },{ 2.0, 0.0, -2.0 },{ 1.0, 0.0, -1.0 } };
	double vFilter[3][3] = { { 1.0, 2.0, 1.0 },{ 0.0, 0.0, 0.0 },{ -1.0, -2.0, -1.0 } };
	Matrix hSobel = createMatrixFromArray(&hFilter[0][0], 3, 3);
	Matrix vSobel = createMatrixFromArray(&vFilter[0][0], 3, 3);
	Matrix inputMx = image2Matrix(img);
	Matrix hResult = convolve(inputMx, hSobel);
	Matrix vResult = convolve(inputMx, vSobel);
	Matrix result = createMatrix(inputMx.height, inputMx.width);
	Image output;
	int i, j;

	for (i = 0; i < inputMx.height; i++)
		for (j = 0; j < inputMx.width; j++)
			result.map[i][j] = sqrt(SQR(hResult.map[i][j]) + SQR(vResult.map[i][j]));

	output = matrix2Image(result, 1, 1.0);

	deleteMatrix(hSobel);
	deleteMatrix(vSobel);
	deleteMatrix(inputMx);
	deleteMatrix(hResult);
	deleteMatrix(vResult);
	deleteMatrix(result);
	return output;
}


// Create and return a Gaussian filter of size (vSize, hSize) with standad deviation sigma.
// The entries in the filter always add up to 1.
Matrix makeGaussianFilter(int vSize, int hSize, double sigma)
{
	double sum = 0.0;
	int i, j;
	Matrix gauss = createMatrix(vSize, hSize);

	for (i = 0; i < vSize; i++)
		for (j = 0; j < hSize; j++)
		{
			gauss.map[i][j] = exp(-(SQR((double)i - (double)(vSize - 1) / 2.0) + SQR((double)j - (double)(hSize - 1) / 2.0)) / SQR(sigma) / 2.0);
			sum += gauss.map[i][j];
		}
	for (i = 0; i < vSize; i++)
		for (j = 0; j < hSize; j++)
			gauss.map[i][j] /= sum;
	return gauss;
}


Image gauss(Image img, int size)
{
	Matrix gFilter = makeGaussianFilter(size, size, (double)size / 4.0);
	Matrix inputMx = image2Matrix(img);
	Matrix result = convolve(inputMx, gFilter);
	Image output = matrix2Image(result, 0, 1.0);

	deleteMatrix(gFilter);
	deleteMatrix(inputMx);
	deleteMatrix(result);
	return output;
}


void main()
{
	Image inputImg = readImage("sample.ppm");
	Image sobelImg = sobel(inputImg);
	Image gauss3Img = gauss(inputImg, 3);
	Image gauss9Img = gauss(inputImg, 9);
	
	writeImage(sobelImg, "sobel.pgm");
	writeImage(gauss3Img, "gauss3.pgm");
	writeImage(gauss9Img, "gauss9.pgm");
	
	deleteImage(inputImg);
	deleteImage(sobelImg);
	deleteImage(gauss3Img);
	deleteImage(gauss9Img);
}

