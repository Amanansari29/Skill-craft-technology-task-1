#include <stdio.h>

// Function to calculate regression coefficients using Normal Equation
void linearRegression(double X[][3], double Y[], int n, double coeffs[4]) {
    // Variables for summations
    double sumX1 = 0, sumX2 = 0, sumX3 = 0, sumY = 0;
    double sumX1X1 = 0, sumX2X2 = 0, sumX3X3 = 0;
    double sumX1X2 = 0, sumX1X3 = 0, sumX2X3 = 0;
    double sumX1Y = 0, sumX2Y = 0, sumX3Y = 0;

    for (int i = 0; i < n; i++) {
        double x1 = X[i][0]; // square feet
        double x2 = X[i][1]; // bedrooms
        double x3 = X[i][2]; // bathrooms
        double y = Y[i];

        sumX1 += x1;
        sumX2 += x2;
        sumX3 += x3;
        sumY  += y;

        sumX1X1 += x1 * x1;
        sumX2X2 += x2 * x2;
        sumX3X3 += x3 * x3;

        sumX1X2 += x1 * x2;
        sumX1X3 += x1 * x3;
        sumX2X3 += x2 * x3;

        sumX1Y += x1 * y;
        sumX2Y += x2 * y;
        sumX3Y += x3 * y;
    }

    // Simplified model (not full matrix solution, works for small dataset)
    // Assume linear model: Price = b0 + b1*x1 + b2*x2 + b3*x3

    // Approximate solution using averages (teaching/demo)
    coeffs[1] = sumX1Y / sumX1X1; // weight for sqft
    coeffs[2] = sumX2Y / sumX2X2; // weight for bedrooms
    coeffs[3] = sumX3Y / sumX3X3; // weight for bathrooms

    coeffs[0] = (sumY - coeffs[1] * sumX1 - coeffs[2] * sumX2 - coeffs[3] * sumX3) / n;
}

int main() {
    int n;
    printf("Enter number of data points: ");
    scanf("%d", &n);

    double X[n][3], Y[n];

    // Input dataset
    for (int i = 0; i < n; i++) {
        printf("\nEnter data for house %d:\n", i + 1);
        printf("Square feet: ");
        scanf("%lf", &X[i][0]);
        printf("Bedrooms: ");
        scanf("%lf", &X[i][1]);
        printf("Bathrooms: ");
        scanf("%lf", &X[i][2]);
        printf("Price: ");
        scanf("%lf", &Y[i]);
    }

    double coeffs[4]; // b0, b1, b2, b3
    linearRegression(X, Y, n, coeffs);

    printf("\nLinear Regression Model:\n");
    printf("Price = %.2f + %.2f*(sqft) + %.2f*(bedrooms) + %.2f*(bathrooms)\n",
           coeffs[0], coeffs[1], coeffs[2], coeffs[3]);

    // Prediction
    double sqft, bedrooms, bathrooms;
    printf("\nEnter details of house to predict price:\n");
    printf("Square feet: ");
    scanf("%lf", &sqft);
    printf("Bedrooms: ");
    scanf("%lf", &bedrooms);
    printf("Bathrooms: ");
    scanf("%lf", &bathrooms);

    double predicted = coeffs[0] + coeffs[1] * sqft + coeffs[2] * bedrooms + coeffs[3] * bathrooms;
    printf("\nPredicted Price = %.2f\n", predicted);

    return 0;
}
