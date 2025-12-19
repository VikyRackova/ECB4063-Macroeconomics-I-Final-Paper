To replicate the full analysis, please follow the steps below.

1. Data manipulation

The file Data_manipulation.jl contains the code used to construct the final dataset.
This script loads the individual macroeconomic data series, selects the relevant variables, and deflates all nominal variables using the GDP deflator (base year 2015). 
Once all variables are expressed in real terms, the countries of interest are selected and aggregated into a single EU-wide macroeconomy by computing annual averages across countries.

If you prefer to skip the data construction step, you may directly use the pre-processed dataset Aggregated_Data.csv, which is the output of the data manipulation procedure, and proceed straight to the analysis.

2. Analysis

The main analysis is implemented in Analysis.jl. 
This script begins with exploratory data analysis, including plots of key macroeconomic ratios over time and the estimation of selected parameters using OLS. 
It then proceeds with the calibration of the model and the computation of regime-dependent steady states.
The dynamic analysis includes impulse response functions to public investment shocks, followed by the computation of fiscal multipliers. 
As a robustness check, impulse response functions to total factor productivity (TFP) shocks are also analyzed. The analysis concludes with a Simulated Method of Moments (SMM) exercise to further assess the model’s performance.

3. Results

All results, together with a detailed description of the research question, methodology, and findings, are presented in the file Final_Paper.pdf.

If you have any questions regarding the analysis or notice any errors, please do not hesitate to contact us.
