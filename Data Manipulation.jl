using CSV
using DataFrames
using GLM
using Statistics
using Plots
using Latexify
using Dates
using StatsPlots
#using Pkg
#Pkg.add("StatsModels")
using StatsModels
using PlotlyJS


########################################################################################################################
####################################### Load all individual time series ################################################
########################################################################################################################
Nominal_GDP = CSV.read("Nominal GDP (Current prices,million EUR).csv", DataFrame)
Nominal_GDP = select(Nominal_GDP,:geo,:TIME_PERIOD,:OBS_VALUE)
rename!(Nominal_GDP,[:Country,:Year,:GDP])

Deflator = CSV.read("Price index (implicit deflator), 2015=100, euro.csv", DataFrame)
Deflator = select(Deflator,:geo,:TIME_PERIOD,:OBS_VALUE)
rename!(Deflator,[:Country,:Year,:Deflator])

Consumption= CSV.read("Final consumption expenditure of households.csv", DataFrame)
Consumption  = filter(row -> row.unit == "Current prices, million euro", Consumption)
Consumption = select(Consumption,:geo,:TIME_PERIOD,:OBS_VALUE)
rename!(Consumption,[:Country,:Year,:Consumption])

Debt= CSV.read("General government consolidated gross debt Million EUR.csv", DataFrame)
Debt = select(Debt,:geo,:TIME_PERIOD,:OBS_VALUE)
rename!(Debt,[:Country,:Year,:Debt])

Public_Investment= CSV.read("General government gross fixed capital formation (million EUR).csv", DataFrame)
Public_Investment = select(Public_Investment,:geo,:TIME_PERIOD,:OBS_VALUE)
rename!(Public_Investment,[:Country,:Year,:Public_Investment])


Taxes= CSV.read("General government total revenue (million EUR).csv", DataFrame)
Taxes = select(Taxes,:geo,:TIME_PERIOD,:OBS_VALUE)
rename!(Taxes,[:Country,:Year,:Taxes])

Private_Investment= CSV.read("Gross fixed capital formation (million EUR).csv", DataFrame)
Private_Investment = select(Private_Investment,:geo,:TIME_PERIOD,:OBS_VALUE)
rename!(Private_Investment,[:Country,:Year,:Private_Investment])

Wages = CSV.read("Compensation of employees (%GDP and Million EUR).csv", DataFrame)
Wages  = filter(row -> row.unit == "Current prices, million euro", Wages)
Wages  = select(Wages,:geo,:TIME_PERIOD,:OBS_VALUE)
rename!(Wages,[:Country,:Year,:Wages])
 
Private_Capital = CSV.read("Public and private capital(%GDP).csv", DataFrame)
Private_Capital  = filter(row -> row.INDICATOR == "Capital stock, Private sector, Constant prices, Percent of GDP", Private_Capital)
Private_Capital  = select(Private_Capital,:COUNTRY,:TIME_PERIOD,:OBS_VALUE)
rename!(Private_Capital,[:Country,:Year,:Private_Capital])

Public_Capital = CSV.read("Public and private capital(%GDP).csv", DataFrame)
Public_Capital  = filter(row -> row.INDICATOR == "Capital stock, General government, Constant prices, Percent of GDP", Public_Capital)
Public_Capital  = select(Public_Capital,:COUNTRY,:TIME_PERIOD,:OBS_VALUE)
rename!(Public_Capital,[:Country,:Year,:Public_Capital])

################################# Forecast values for 2020-2024  ###############################################################
function forecast_by_country(df::DataFrame, value_col::Symbol;
                             last_year::Int = 2019,
                             future_years = 2020:2024)

    forecasts = DataFrame[]

    for sub in groupby(df, :Country)
        # use data up to last_year
        hist = filter(row -> row.Year ≤ last_year && !ismissing(row[value_col]), sub)

        if nrow(hist) < 3
            @warn "Too few observations for $(hist.Country[1]) in $(value_col)"
            continue
        end

        # rename value column to :y for the regression
        hist_reg = select(hist, :Country, :Year, value_col => :y)

        # linear trend: y ~ Year
        model = lm(@formula(y ~ Year), hist_reg)

        # future years
        fut = DataFrame(Year = collect(future_years))
        fut.y_hat = predict(model, fut)

        fut.Country = fill(hist.Country[1], nrow(fut))
        fut[!, value_col] = fut.y_hat
        select!(fut, :Country, :Year, value_col)

        push!(forecasts, fut)
    end

    return vcat(forecasts...)
end

Private_Capital_forecast = forecast_by_country(Private_Capital, :Private_Capital)
Public_Capital_forecast  = forecast_by_country(Public_Capital,  :Public_Capital)

# Combine historical + forecasts (still in two separate data frames)
Private_Capital_full = vcat(Private_Capital, Private_Capital_forecast)
Public_Capital_full  = vcat(Public_Capital,  Public_Capital_forecast)
replace!(Private_Capital_full.Country, "Slovenia,Republic of" => "Slovenia")
replace!(Public_Capital_full.Country, "Slovenia,Republic of" => "Slovenia")
replace!(Private_Capital_full.Country, "Croatia,Republic of" => "Croatia")
replace!(Public_Capital_full.Country, "Croatia,Republic of" => "Croatia")
replace!(Private_Capital_full.Country, "Slovak Republic" => "Slovakia")
replace!(Public_Capital_full.Country, "Slovak Republic" => "Slovakia")
replace!(Private_Capital_full.Country, "Czech Republic" => "Czechia")
replace!(Public_Capital_full.Country, "Czech Republic" => "Czechia")
replace!(Private_Capital_full.Country, "Estonia,Republic of" => "Estonia")
replace!(Public_Capital_full.Country, "Estonia,Republic of" => "Estonia")
replace!(Private_Capital_full.Country, "Lithuania,Republic of" => "Lithuania")
replace!(Public_Capital_full.Country, "Lithuania,Republic of" => "Lithuania")
replace!(Private_Capital_full.Country, "Latvia,Republic of" => "Latvia")
replace!(Public_Capital_full.Country, "Latvia,Republic of" => "Latvia")
replace!(Private_Capital_full.Country, "Netherlands, The" => "Netherlands")
replace!(Public_Capital_full.Country, "Netherlands, The" => "Netherlands")
replace!(Private_Capital_full.Country, "Poland,Republic of" => "Poland")
replace!(Public_Capital_full.Country, "Poland,Republic of" => "Poland")


Interest = CSV.read("Interest rate.csv", DataFrame)
Interest.date = Date.(Interest.DATE, dateformat"m/d/yyyy")   # adjust to actual format
Interest.year = year.(Interest.date)
Interest = combine(groupby(Interest, :year), :OBS_VALUE => mean => :mro_annual)
rename!(Interest,[:Year,:Interest])

Inflation = CSV.read("All-items HICP Annual average rate of change.csv", DataFrame)
Inflation  = select(Inflation,:geo,:TIME_PERIOD,:OBS_VALUE)
rename!(Inflation,[:Country,:Year,:Inflation])

All_variables = [Nominal_GDP,Deflator,Consumption,Debt,Public_Investment,Taxes,Private_Investment, Wages,Private_Capital_full,Public_Capital_full]
Fulldataset = reduce((a,b)-> innerjoin(a,b,on = [:Country,:Year]),All_variables)
Fulldataset = filter(:Country => in(EU_countries), Fulldataset)

Fulldataset.defl_level = Fulldataset.Deflator ./ 100
Fulldataset.Real_GDP              = Fulldataset.GDP ./ Fulldataset.defl_level
Fulldataset.Real_Consumption      = Fulldataset.Consumption ./ Fulldataset.defl_level
Fulldataset.Real_PublicInvestment = Fulldataset.Public_Investment ./ Fulldataset.defl_level
Fulldataset.Real_PrivateInvestment = Fulldataset.Private_Investment ./ Fulldataset.defl_level
Fulldataset.Real_Debt             = Fulldataset.Debt ./ Fulldataset.defl_level
Fulldataset.Real_Taxes            = Fulldataset.Taxes ./ Fulldataset.defl_level
Fulldataset.Real_Wages            = Fulldataset.Wages ./ Fulldataset.defl_level
Fulldataset.Real_PrivateCapital            = (Fulldataset.Private_Capital ./ 100) .* Fulldataset.Real_GDP
Fulldataset.Real_PublicCapital            = (Fulldataset.Public_Capital ./ 100) .* Fulldataset.Real_GDP


Real_variables = Fulldataset
Real_variables  = select(Real_variables,:Country,:Year,:Real_GDP,:Real_Consumption,:Real_PublicInvestment,:Real_PrivateInvestment,:Real_Debt,:Real_Taxes,:Real_Wages, :Real_PrivateCapital,:Real_PublicCapital)
rename!(Real_variables,[:Country,:Year,:GDP,:Consumption, :Public_Investment,:Private_Investment,:Debt,:Taxes,:Wages,:Private_Capital,:Public_Capital])

sort!(Real_variables, [:Country, :Year])
CSV.write("All real variables.csv",Real_variables)

infl_annual = combine(groupby(Inflation, :Year),
                      :Inflation => mean => :Inflation_avg)

Interst_Inflation = innerjoin(Interest, infl_annual, on = :Year)

# 3. Convert to decimals and compute real rate (Fisher)
Interst_Inflation.i  = Interst_Inflation.Interest ./ 100
Interst_Inflation.pi = Interst_Inflation.Inflation_avg ./ 100
Interst_Inflation.r_real = (1 .+ Interst_Inflation.i) ./ (1 .+ Interst_Inflation.pi) .- 1

Real_interest_rate = Interst_Inflation
Real_interest_rate  = select(Real_interest_rate,:Year,:r_real)
rename!(Real_interest_rate,[:Year,:Interest])

start_year = 1995
end_year   = maximum(Real_interest_rate.Year)
full_years = start_year:end_year
full_df = DataFrame(Year = full_years)

# Join the interest-rate change years
full_df = leftjoin(full_df, Real_interest_rate, on = :Year)

# ---- 2) Set 1995–1998 equal to 1999 rate ----
first_year  = minimum(Real_interest_rate.Year)                 # 1999
first_rate  = Real_interest_rate.Interest[argmin(Real_interest_rate.Year)]     # interest in 1999

# all years before first_year get the first_rate
idx_pre = findall(<(first_year), full_df.Year) # Years < 1999
full_df.Interest[idx_pre] .= first_rate

# ---- 3) Forward-fill for gaps between change years ----
for i in 2:nrow(full_df)
    if ismissing(full_df.Interest[i])
        full_df.Interest[i] = full_df.Interest[i-1]
    end
end

sort!(full_df, [ :Year])
Real_interest_rate_full = full_df
CSV.write("Real interest rate.csv",Real_interest_rate_full)

Hoursworked = CSV.read("Hours worked per employed person.csv", DataFrame)
Hoursworked= select(Hoursworked,:geo,:TIME_PERIOD,:OBS_VALUE)
rename!(Hoursworked,[:Country,:Year,:Labour])


########################################################################################################################
##################################### Select the countries of interest and create an aggregated EU Economy #############
########################################################################################################################
EU_countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus",
    "Czechia", "Denmark", "Estonia", "Finland", "France",
    "Germany", "Greece", "Hungary", "Ireland", "Italy",
    "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania",
    "Slovakia", "Slovenia", "Spain", "Sweden"
]

labourdata = Hoursworked
labourdata = filter(:Country => in(EU_countries), labourdata)

fullfull = innerjoin(Real_variables,labourdata, on = [:Country,:Year])
  
num_cols = names(fullfull, col -> eltype(fullfull[!, col]) <: Union{Missing, Number})
num_cols = setdiff(num_cols, [:Year]) 

# average per year across all countries
Aggregated_Dataset = combine(
    groupby(fullfull, :Year),
    num_cols .=> (x -> mean((x))) .=> num_cols,
)
Aggregated_Dataset.DebttoGDP = Aggregated_Dataset.Debt./Aggregated_Dataset.GDP

# Save the transformed series into a csv to work with it later
Readytocalibratedataset = innerjoin(Aggregated_Dataset,Real_interest_rate_full, on = :Year)
CSV.write("Aggregated_Data.csv",Readytocalibratedataset)

#####################################################################################################################################
################################# Plot the debt to gdp of the selected countries ##################################
#####################################################################################################################################
Debt_GDP_EU = filter(:Country => in(EU_countries), Real_variables)
Debt_GDP_EU = select(Debt_GDP_EU,:Country,:Year,:Debt,:GDP)
Debt_GDP_EU.DebttoGDP = Debt_GDP_EU.Debt./Debt_GDP_EU.GDP

Agg = select(Aggregated_Dataset, :Year, :DebttoGDP)  # adjust col names if needed
sort!(Debt_GDP_EU, [:Country, :Year])
sort!(Agg, :Year)

# 3) Base plot: all countries in grey tones
plt = plot(
    xlabel = "Year",
    ylabel = "Debt-to-GDP",
    title  = "Debt-to-GDP Ratios in the EU Countries",
    legend = :outerbottom,
    size   = (800, 500),
    bottom_margin = 10Plots.mm,
    left_margin   = 10Plots.mm,
    top_margin    = 2Plots.mm,
    xlims  = (1995, 2024)
)

# Plot each country in grey (no legend entries)
for c in EU_countries
    dfc = Debt_GDP_EU[Debt_GDP_EU.Country .== c, :]
    plot!(plt, dfc.Year, dfc.DebttoGDP;
          color = :grey60, alpha = 0.6, lw = 1.5, label = "")
end

# Overlay aggregate in a strong color (single legend entry)
plot!(plt, Agg.Year, Agg.DebttoGDP; legend = (0.75, -0.15),
      color = :blue, lw = 3, label = "Aggregate (EU)")


# Plot a map of countries of interest
values = ones(length(EU_countries))
trace = choropleth(
    locations    = EU_countries,
    locationmode = "country names",
    z            = values,
    colorscale   = [[0.0, "royalblue"], [1.0, "royalblue"]],
    showscale    = false           # hide colour bar
)

layout = Layout(
    title = "Countries in our sample",
    geo   = attr(
        scope        = "europe",
        projection   = attr(type="natural earth"),
        showcountries = true,
        lataxis = attr(range = [32, 70]),
        lonaxis = attr(range = [-9, 28])
    )
)

plt = Plot(trace, layout)
display(plt)

