using CSV
using DataFrames
using GLM
using Statistics
using Plots
using Latexify
using Dates
using Random
using StatsPlots
using PlutoUI, Roots, LinearAlgebra, NLsolve, NonlinearSolve,
      Plots, Optim, DataFrames, QuantEcon, Interpolations
#using Pkg
#Pkg.add("SciMLBase")
using StatsModels
using SciMLBase   

########################################################################################################################
################################### Load the Dataset 
########################################################################################################################
Dataset = CSV.read("Aggregated_Data.csv", DataFrame)
########################################################################################################################
################################### Plot the ratios to confirm their stationarity over time
########################################################################################################################
Ratios = select(Dataset, Not([:Labour, :Interest, :Wages,:DebttoGDP,:Debt]))
Ratios = Ratios[:, Not([:GDP, :Year])] ./ Dataset.GDP
Ratios = hcat(Dataset[:, [:Year, :GDP]], Ratios)

ratio_cols = names(Ratios, Not([:Year, :GDP,]))
p = plot(legend = :outerbottom,legendcols  = length(ratio_cols), size=(1000,300))  

for c in ratio_cols
    plot!(p,
          Ratios.Year,         
          Ratios[!, c],         
          label = String(c))   
end

xlabel!("Year")
ylabel!("Ratio to GDP")
title!("All Ratios over Time")
########################################################################################################################
################################### Categorize the data into low and high debt period
########################################################################################################################
Dataset.DebtRegime = ifelse.(Dataset.Year .<= 2007, "low", "high")

########################################################################################################################
################################## Compute the Solow residual
########################################################################################################################
function compute_log_tfp(Y, K, N, α)
   
    Yv = Vector{Float64}(Y)
    Kv = Vector{Float64}(K)
    Nv = Vector{Float64}(N)

    log_z = log.(Yv) .- α .* log.(Kv) .- (1.0 - α) .* log.(Nv)
    return log_z
end

function estimate_ar1(y; include_const::Bool=true, include_trend::Bool=false)
    yv = Vector{Float64}(y)
    T  = length(yv)
    @assert T >= 3 

    y_t   = yv[2:end]
    y_lag = yv[1:end-1]

   
    cols = Vector{Vector{Float64}}()
    if include_const
        push!(cols, ones(T-1))
    end
    if include_trend
        
        push!(cols, collect(2:T) .|> Float64)
    end
    push!(cols, y_lag)

    X = hcat(cols...)  
    β = X \ y_t        

   
    idx = 1
    ĉ = include_const ? β[idx] : 0.0
    idx += include_const ? 1 : 0
    γ̂ = include_trend ? β[idx] : 0.0
    idx += include_trend ? 1 : 0
    ρ̂ = β[end]

    residuals = y_t .- X * β
    σ̂ = std(residuals)

    return (ρ=ρ̂, σ=σ̂, c=ĉ, γ=γ̂, residuals=residuals, y_used=yv)
end

function estimate_tfp_ar1(Y, K, N, α; include_const::Bool=true, include_trend::Bool=false)
    log_tfp = compute_log_tfp(Y, K, N, α)
    ar1 = estimate_ar1(log_tfp; include_const=include_const, include_trend=include_trend)
    return merge(ar1, (log_tfp=log_tfp,))
end

tfp_ar1 = estimate_tfp_ar1(Dataset.GDP, Dataset.Private_Capital, Dataset.Labour, 0.33;
                           include_const=true, include_trend=false) # here we tried with both trend and without trend but without the trend the persistence is too low

println("ρ̂ = ", tfp_ar1.ρ)
println("σ̂ = ", tfp_ar1.σ)

########################################################################################################################
################################### Compute the persistence parameter for the Investment
########################################################################################################################
df = copy(Dataset)
df.LogIG = log.(df.Public_Investment)

results = DataFrame(DebtRegime=String[], rho_g=Float64[], sigma_g=Float64[], n=Int[])

for sdf in groupby(df, :DebtRegime)
    sdf = sort(copy(sdf), :Year)

   
    logIGbar = mean(sdf.LogIG)
    sdf.g = sdf.LogIG .- logIGbar

    
    sdf.gLag = [missing; sdf.g[1:end-1]]
    sdf = dropmissing(sdf, [:g, :gLag])

    
    m = lm(@formula(g ~ 0 + gLag), sdf)
    ρ = coef(m)[1]
    σ = std(residuals(m))

    push!(results, (string(first(sdf.DebtRegime)), ρ, σ, nrow(sdf)))
    println("\nRegime: ", first(sdf.DebtRegime))
    println(m)
    println("sigma_g = ", σ)
end

println("\nSummary:")
println(results)

########################################################################################################################
###################################  Compute the multipliers in the fiscal rule 
########################################################################################################################
df = copy(Dataset)

df.Tau  = df.Taxes ./ df.GDP
df.LogY = log.(df.GDP)

df = dropmissing(df, [:Tau, :DebttoGDP, :GDP, :Year, :DebtRegime])

function prep_regime!(sdf::DataFrame)
   
    bbar   = mean(sdf.DebttoGDP)
    taubar = mean(sdf.Tau)

    
    X = hcat(ones(nrow(sdf)), sdf.Year)
    β = X \ sdf.LogY
    logYtrend = X * β

    sdf.BGap  = sdf.DebttoGDP .- bbar
    sdf.YGap  = sdf.LogY .- logYtrend
    sdf.TauDm = sdf.Tau  .- taubar
    return sdf
end

results = DataFrame(DebtRegime=String[], phi_b=Float64[], phi_y=Float64[], n=Int[])

for sdf in groupby(df, :DebtRegime)
    sdf = prep_regime!(copy(sdf))
    m = lm(@formula(TauDm ~ 0 + BGap + YGap), sdf)

    push!(results, (
        DebtRegime = string(first(sdf.DebtRegime)),
        phi_b      = coef(m)[1],
        phi_y      = coef(m)[2],
        n          = nrow(sdf)
    ))

    println("\n==================== Regime: ", first(sdf.DebtRegime), " ====================")
    println(m)
end

println("\nSummary coefficients:")
println(results)
########################################################################################################################
################################## Calibrate the Parameters for steady state
########################################################################################################################

calib_moments = combine(
    groupby(Dataset, :DebtRegime)
) do sdf
    DataFrame(
        DebtRegime = first(sdf.DebtRegime),
        IP_Y   = mean(sdf.Private_Investment ./ sdf.GDP),
        IG_Y   = mean(sdf.Public_Investment  ./ sdf.GDP),
        Kp_Y   = mean(sdf.Private_Capital    ./ sdf.GDP),
        Kg_Y   = mean(sdf.Public_Capital     ./ sdf.GDP),
        B_Y    = mean(sdf.DebttoGDP),
        rbar   = 0.04,
        Nbar   = 1/3
    )
end

################################################################################
# Base parameters - same in both regimes
################################################################################

param_base = (
    β  = 0.96,     # annual discount factor
    α  = 0.33,     # private capital 
    σ = 2.0,        # Risk aversion
    αG = 0.05,     # public capital share
    φ  = 1.0,      # Frisch elasticity parameter
    A  = 1.0       # TFP normalization
)

################################################################################
# Steady state specific parameters
################################################################################
function build_param_ss(regime::AbstractString,
                        calib_moments::DataFrame,
                        param_base)

    row = calib_moments[calib_moments.DebtRegime .== regime, :]
    @assert nrow(row) == 1

    # Targets from data
    N̄   = row.Nbar[1]
    b̄   = row.B_Y[1]
    igy = row.IG_Y[1]
    i   = row.rbar[1]

    # Depreciation rates
    δ  = row.IP_Y[1] / row.Kp_Y[1]
    δG = row.IG_Y[1] / row.Kg_Y[1]

    return merge(param_base, (
        δ     = δ,
        δG    = δG,
        b̄     = b̄,
        igy   = igy,
        i     = i,
        N̄     = N̄
    ))
end

################################################################################
# STEADY-STATE EQUATIONS 
################################################################################
function rbc_equation_ss!(F, x, p)

    C, I, N, K, Y, KG, IG, B, ψ = x

    if any(v -> v <= 0, x)
        F .= 1e6
        return
    end

    r = p.α * Y / K
    w = (1 - p.α - p.αG) * Y / N

    # (1) Euler
    F[1] = 1.0 - p.β * (1.0 + r - p.δ)

    # (2) Target labor directly
    F[2] = N - p.N̄

    # (3) Production
    F[3] = Y - p.A * K^p.α * KG^p.αG * N^(1 - p.α - p.αG)

    # (4) Resource constraint
    F[4] = Y - (C + I + IG + p.i * B)

    # (5) Public capital accumulation
    F[5] = KG - IG / p.δG

    # (6) Debt target
    F[6] = B - p.b̄ * Y

    # (7) Private investment
    F[7] = I - p.δ * K

    # (8) Public investment share
    F[8] = IG - p.igy * Y

    # (9) GHH labor FOC (now pins down ψ)
    F[9] = ψ * N^p.φ - w
end

################################################################################
# SOLVE STEADY STATE 
################################################################################
function solve_ss_for_regime(regime::AbstractString,
                             calib_moments::DataFrame,
                             param_base,
                             x0)

    p = build_param_ss(regime, calib_moments, param_base)

    prob = NonlinearProblem(
        rbc_equation_ss!,
        x0,
        p,
        isoutofdomain = (x, _) -> any(v -> v ≤ 0, x)
    )

    sol = NonlinearSolve.solve(prob)
    @assert sol.retcode == ReturnCode.Success "Steady state failed for regime $regime"

    C, I, N, K, Y, KG, IG, B, ψ = sol.u
    T  = IG + p.i * B
    τ  = T / Y

    return p, (; C, I, N, K, Y, KG, IG, B, T, τ, ψ)
end

x0 = [0.7, 0.2, 0.33, 3.0, 1.0, 2.0, 0.05, 0.6,2.0]

param_low,  ss_low  = solve_ss_for_regime("low",  calib_moments, param_base, x0)
param_high, ss_high = solve_ss_for_regime("high", calib_moments, param_base, x0)
################################################################################
# PRINT THE RESULTS
################################################################################
ss_table = DataFrame(
    Variable = [
        "Y (Output)",
        "C (Consumption)",
        "I (Private investment)",
        "I_G (Public investment)",
        "N (Labour)",
        "K (Private capital)",
        "K_G (Public capital)",
        "B (Debt)",
        "T (Taxes)",
        "τ (Tax rate)",
        "B/Y",
        "I_G/Y",
        "I/Y",
        "K_G/Y",
        "K/Y",
        "C/Y",
        "i·B/Y",
        "ψ"
    ],
    LowDebt = [
        ss_low.Y,
        ss_low.C,
        ss_low.I,
        ss_low.IG,
        ss_low.N,
        ss_low.K,
        ss_low.KG,
        ss_low.B,
        ss_low.T,
        ss_low.τ,
        ss_low.B / ss_low.Y,
        ss_low.IG / ss_low.Y,
        ss_low.I / ss_low.Y,
        ss_low.KG / ss_low.Y,
        ss_low.K / ss_low.Y,
        ss_low.C / ss_low.Y,
        param_low.i * (ss_low.B / ss_low.Y),
        ss_low.ψ,
    ],
    HighDebt = [
        ss_high.Y,
        ss_high.C,
        ss_high.I,
        ss_high.IG,
        ss_high.N,
        ss_high.K,
        ss_high.KG,
        ss_high.B,
        ss_high.T,
        ss_high.τ,
        ss_high.B / ss_high.Y,
        ss_high.IG / ss_high.Y,
        ss_high.I / ss_high.Y,
        ss_high.KG / ss_high.Y,
        ss_high.K / ss_high.Y,
        ss_high.C / ss_high.Y,
        param_high.i * (ss_high.B / ss_high.Y),
        ss_high.ψ
    ]
)

latex_ss = latexify(
    ss_table,
    env = :table,
    fmt = "%.4f",
    latex = false
)


function param_table(p_low::NamedTuple, p_high::NamedTuple)
    common_keys = intersect(keys(p_low), keys(p_high))

    df = DataFrame(
        Parameter = String[],
        LowDebt   = Float64[],
        HighDebt  = Float64[]
    )

    for k in sort(collect(common_keys); by=string)
        vlow  = getproperty(p_low, k)
        vhigh = getproperty(p_high, k)

        if (vlow isa Real) && (vhigh isa Real)
            push!(df, (string(k), float(vlow), float(vhigh)))
        end
    end

    return df
end

df_params = param_table(param_low, param_high)

latex_params = latexify(df_params, env = :table, fmt = "%.4f", latex = false)




########################################################################################################################
################################## Derive the Impulse response functions for public investment
########################################################################################################################
function irf_public_investment_pf_logI(p::NamedTuple, ss::NamedTuple;
        T::Int=40,
        shock_size::Float64=0.01,
        ρg::Float64=0.6,
        φb::Float64=0.05,
        φy::Float64=0.0,
        τ_min::Float64=0.0,
        τ_max::Float64=0.95,
        ϵ::Float64=1e-12
    )

    @assert T >= 5

    # --- parameters ---
    β  = p.β
    σ  = p.σ
    α  = p.α
    αG = p.αG
    δ  = p.δ
    δG = p.δG
    φ  = p.φ
    A  = p.A
    i  = p.i
    b̄  = p.b̄

    # --- steady state ---
    Ȳ  = ss.Y
    C̄  = ss.C
    N̄  = ss.N
    K̄  = ss.K
    KḠ = ss.KG
    IḠ = ss.IG
    B̄  = ss.B
    τ̄  = ss.τ
    ψ   = ss.ψ
    Ī  = ss.I

    
    compC(C, N) = C - ψ * N^(1 + φ) / (1 + φ)

   
    function labor_GHH(K, KG)
        num = (1 - α - αG) * A * (K^α) * (KG^αG)
        return (num / ψ)^(1 / (φ + α + αG))
    end

    prod(K, KG, N) = A * (K^α) * (KG^αG) * (N^(1 - α - αG))

    function taxrate(B, Y)
        Ysafe = max(Y, ϵ)
        b = B / Ysafe
        gap = log(Ysafe) - log(max(Ȳ, ϵ))
        τ_raw = τ̄ + φb * (b - b̄) + φy * gap
        return clamp(τ_raw, τ_min, τ_max)
    end

   
    g = zeros(T)
    g[1] = log(1 + shock_size)
    for t in 2:T
        g[t] = ρg * g[t-1]
    end
    IG = IḠ .* exp.(g)

    
    KG = zeros(T)
    KG[1] = KḠ
    for t in 1:(T-1)
        KG[t+1] = (1 - δG) * KG[t] + IG[t]
    end

    λB = 0.01

  
    z0 = fill(log(max(Ī, ϵ)), T-1)

    function residuals!(F::AbstractVector, z::AbstractVector)
        @assert length(F) == T-1
        @assert length(z) == T-1

       
        I = exp.(z)  
        K = zeros(T)
        K[1] = K̄
        for t in 1:(T-1)
            K[t+1] = (1 - δ) * K[t] + I[t]
        end

        
        if any(.!isfinite.(K)) || any(K .<= ϵ) || any(.!isfinite.(I))
            F .= 1e6
            return
        end
        if any(.!isfinite.(KG)) || any(KG .<= ϵ)
            F .= 1e6
            return
        end

        # simulate
        B = zeros(T); B[1] = B̄
        C = zeros(T); N = zeros(T); Y = zeros(T); τ = zeros(T); Ttax = zeros(T)
        r = zeros(T)

        for t in 1:T
            N[t] = labor_GHH(K[t], KG[t])
            if !isfinite(N[t]) || N[t] <= 0
                F .= 1e6; return
            end

            Y[t] = prod(K[t], KG[t], N[t])
            if !isfinite(Y[t]) || Y[t] <= 0
                F .= 1e6; return
            end

            r[t] = α * Y[t] / K[t]
            if !isfinite(r[t])
                F .= 1e6; return
            end

            τ[t] = taxrate(B[t], Y[t])
            Ttax[t] = τ[t] * Y[t]

            I_t = (t <= T-1) ? I[t] : δ * K[t]

            
            C[t] = Y[t] - I_t - IG[t] - i * B[t]
            if !isfinite(C[t]) || C[t] <= 0
                F .= 1e6; return
            end

            X = compC(C[t], N[t])
            if !isfinite(X) || X <= 0
                F .= 1e6; return
            end

            if t < T
                B[t+1] = IG[t] + (1 + i) * B[t] - Ttax[t]
                if !isfinite(B[t+1])
                    F .= 1e6; return
                end
            end
        end

        
        for t in 1:(T-2)
            X_t   = compC(C[t],   N[t])
            X_tp1 = compC(C[t+1], N[t+1])
            F[t] = X_t^(-σ) - β * X_tp1^(-σ) * (1 + r[t+1] - δ)
        end

        
        F[T-1] = (K[T] - K̄) + λB * (B[T] - B̄)

        return
    end

    sol = nlsolve(
        residuals!, z0;
        method = :trust_region,
        xtol = 1e-10,
        ftol = 1e-10,
        iterations = 80_000,
        autoscale = false
    )

    @assert NLsolve.converged(sol) "PF solver did not converge. Try smaller shock_size, smaller ρg, or smaller φb."

    zstar = sol.zero
    I_dec = exp.(zstar)

    
    I = zeros(T)
    I[1:T-1] .= I_dec

    K = zeros(T); K[1] = K̄
    for t in 1:(T-1)
        K[t+1] = (1 - δ) * K[t] + I[t]
    end
    I[T] = δ * K[T]

    B = zeros(T); B[1] = B̄
    C = zeros(T); N = zeros(T); Y = zeros(T); τ = zeros(T); Ttax = zeros(T)
    r = zeros(T)

    for t in 1:T
        N[t] = labor_GHH(K[t], KG[t])
        Y[t] = prod(K[t], KG[t], N[t])
        r[t] = α * Y[t] / K[t]

        τ[t] = taxrate(B[t], Y[t])
        Ttax[t] = τ[t] * Y[t]

        
        C[t] = Y[t] - I[t] - IG[t] - i * B[t]

        if t < T
            B[t+1] = IG[t] + (1 + i) * B[t] - Ttax[t]
        end
    end

    pctlog_pos(x, x̄) = 100 .* log.(max.(x, ϵ) ./ max(x̄, ϵ))
    pctdev(x, x̄)     = 100 .* ((x ./ x̄) .- 1.0)

    return (
        t  = 0:(T-1),
        Y  = pctlog_pos(Y, Ȳ),
        C  = pctlog_pos(C, C̄),
        I  = pctdev(I, Ī),
        N  = pctlog_pos(N, N̄),
        K  = pctlog_pos(K, K̄),
        KG = pctlog_pos(KG, KḠ),
        IG = pctlog_pos(IG, IḠ),
        B  = pctdev(B, B̄),
        τ  = 100 .* (τ .- τ̄)
    )
end

################################################################################
# RUN + PLOT
################################################################################

irf_low  = irf_public_investment_pf_logI(param_low,  ss_low;  T=200, shock_size=0.01, ρg=0.967, φb=0.109277, φy=0.144772)
irf_high = irf_public_investment_pf_logI(param_high, ss_high; T=200, shock_size=0.01,  ρg=0.919, φb=0.104832, φy=0.064843)

function plot_irf_grid_low_high(irf_low, irf_high; main_title="IRF to Public Investment Shock")
    t = irf_low.t

    function twoline(y_low, y_high; ttl="", xlab="", ylab="% dev")
        p = plot(t, y_low,  lw=3, label="Low debt", title=ttl, xlabel=xlab, ylabel=ylab)
        plot!(p, t, y_high, lw=3, ls=:dash, label="High debt")
        hline!(p, [0.0], ls=:dot, label="")
        return p
    end

    p1 = twoline(irf_low.IG, irf_high.IG, ttl="Public Investment (IG)")
    p2 = twoline(irf_low.Y,  irf_high.Y,  ttl="Output (Y)")
    p3 = twoline(irf_low.C,  irf_high.C,  ttl="Consumption (C)")
    p4 = twoline(irf_low.I,  irf_high.I,  ttl="Private Investment (I)", xlab="Periods")
    p5 = twoline(irf_low.KG, irf_high.KG, ttl="Public Capital (KG)",     xlab="Periods")
    p6 = twoline(irf_low.N,  irf_high.N,  ttl="Labor (N)",              xlab="Periods")

    plot(p1, p2, p3, p4, p5, p6,
         layout=(2,3),
         size=(1200,650),
         plot_title=main_title)
end


default(left_margin=10Plots.mm, bottom_margin=8Plots.mm, top_margin=6Plots.mm, right_margin=6Plots.mm)
display(plot_irf_grid_low_high(irf_low, irf_high; main_title="IRF to Public Investment Shock"))

########################################################################################################################
################################## Derive multipliers 
########################################################################################################################
function fiscal_multipliers_diagnostics(irf, ss; shock_size, ρg, H=20)
    T = length(irf.t)

    Y_path = ss.Y .* exp.(irf.Y ./ 100)

    g = zeros(T)
    g[1] = log(1 + shock_size)
    for t in 2:T
        g[t] = ρg * g[t-1]
    end
    IG_path = ss.IG .* exp.(g)

    ΔY  = Y_path  .- ss.Y
    ΔIG = IG_path .- ss.IG

    # Impact 
    m0 = ΔY[1] / ΔIG[1]

    # One-year-ahead 
    m1 = ΔY[2] / ΔIG[1]

    # Peak multiplier 
    mpeek = maximum(ΔY ./ ΔIG[1])

    # Cumulative multiplier
    Hcap = min(H, T)
    mH = sum(ΔY[1:Hcap]) / sum(ΔIG[1:Hcap])

    return (impact=m0, one_year_ahead=m1, peak=mpeek, cumulative=mH, H=Hcap,
            ΔY1=ΔY[1], ΔY2=ΔY[2], ΔIG1=ΔIG[1])
end

m_low  = fiscal_multipliers_diagnostics(irf_low,  ss_low;  shock_size=0.01, ρg=0.967, H=100)
m_high = fiscal_multipliers_diagnostics(irf_high, ss_high; shock_size=0.01, ρg=0.919, H=100)

@show m_low
@show m_high

multipliers = (; HighDebt = m_high, LowDebt = m_low)


metric_keys = [
    (:impact,         "Impact multiplier"),
    (:one_year_ahead, "1-year-ahead multiplier"),
    (:cumulative,     "Cumulative multiplier (0..H)"),
    (:peak,           "Peak multiplier")
]

fmt(x; d=3) = x isa Number ? round(x, digits=d) : x

df = DataFrame(
    Metric   = [k[2] for k in metric_keys],
    LowDebt  = [fmt(getproperty(m_low,  k[1])) for k in metric_keys],
    HighDebt = [fmt(getproperty(m_high, k[1])) for k in metric_keys],
)

pretty_table(df; backend = Val(:latex), tf = tf_latex_booktabs)
########################################################################################################################
################################## Derive the Impulse response functions for TFP shocks
########################################################################################################################
function irf_tfp_shock_pf_logI(p::NamedTuple, ss::NamedTuple;
        T::Int=40,
        shock_size::Float64=0.01,   
        ρa::Float64=0.6,            
        φb::Float64=0.05,
        φy::Float64=0.0,
        τ_min::Float64=0.0,
        τ_max::Float64=0.95,
        ϵ::Float64=1e-12
    )

    @assert T >= 5

    # --- parameters ---
    β  = p.β
    σ  = p.σ
    α  = p.α
    αG = p.αG
    δ  = p.δ
    δG = p.δG
    φ  = p.φ
    Ā  = p.A
    i  = p.i
    b̄  = p.b̄

    # --- steady state ---
    Ȳ  = ss.Y
    C̄  = ss.C
    N̄  = ss.N
    K̄  = ss.K
    KḠ = ss.KG
    IḠ = ss.IG
    B̄  = ss.B
    τ̄  = ss.τ
    ψ   = ss.ψ
    Ī  = ss.I

   
    compC(C, N) = C - ψ * N^(1 + φ) / (1 + φ)

    
    function labor_GHH(K, KG, A_t)
        num = (1 - α - αG) * A_t * (K^α) * (KG^αG)
        return (num / ψ)^(1 / (φ + α + αG))
    end

    prod(K, KG, N, A_t) = A_t * (K^α) * (KG^αG) * (N^(1 - α - αG))

    function taxrate(B, Y)
        Ysafe = max(Y, ϵ)
        b = B / Ysafe
        gap = log(Ysafe) - log(max(Ȳ, ϵ))
        τ_raw = τ̄ + φb * (b - b̄) + φy * gap
        return clamp(τ_raw, τ_min, τ_max)
    end

    
    a = zeros(T)
    a[1] = log(1 + shock_size)
    for t in 2:T
        a[t] = ρa * a[t-1]
    end
    A = Ā .* exp.(a)   

    
    IG = fill(IḠ, T)

    
    KG = zeros(T)
    KG[1] = KḠ
    for t in 1:(T-1)
        KG[t+1] = (1 - δG) * KG[t] + IG[t]
    end

    
    λB = 0.01

    
    z0 = fill(log(max(Ī, ϵ)), T-1)

    function residuals!(F::AbstractVector, z::AbstractVector)
        @assert length(F) == T-1
        @assert length(z) == T-1

        
        I = exp.(z)  
        K = zeros(T)
        K[1] = K̄
        for t in 1:(T-1)
            K[t+1] = (1 - δ) * K[t] + I[t]
        end

        
        if any(.!isfinite.(K)) || any(K .<= ϵ) || any(.!isfinite.(I))
            F .= 1e6
            return
        end
        if any(.!isfinite.(KG)) || any(KG .<= ϵ)
            F .= 1e6
            return
        end
        if any(.!isfinite.(A)) || any(A .<= ϵ)
            F .= 1e6
            return
        end

        
        B = zeros(T); B[1] = B̄
        C = zeros(T); N = zeros(T); Y = zeros(T); τ = zeros(T); Ttax = zeros(T)
        r = zeros(T)

        for t in 1:T
            N[t] = labor_GHH(K[t], KG[t], A[t])
            if !isfinite(N[t]) || N[t] <= 0
                F .= 1e6; return
            end

            Y[t] = prod(K[t], KG[t], N[t], A[t])
            if !isfinite(Y[t]) || Y[t] <= 0
                F .= 1e6; return
            end

            r[t] = α * Y[t] / K[t]
            if !isfinite(r[t])
                F .= 1e6; return
            end

            τ[t] = taxrate(B[t], Y[t])
            Ttax[t] = τ[t] * Y[t]     

            I_t = (t <= T-1) ? I[t] : δ * K[t]

            
            C[t] = Y[t] - I_t - IG[t] - i * B[t]
            if !isfinite(C[t]) || C[t] <= 0
                F .= 1e6; return
            end

            X = compC(C[t], N[t])
            if !isfinite(X) || X <= 0
                F .= 1e6; return
            end

            if t < T
                B[t+1] = IG[t] + (1 + i) * B[t] - Ttax[t]
                if !isfinite(B[t+1])
                    F .= 1e6; return
                end
            end
        end

        
        for t in 1:(T-2)
            X_t   = compC(C[t],   N[t])
            X_tp1 = compC(C[t+1], N[t+1])
            F[t] = X_t^(-σ) - β * X_tp1^(-σ) * (1 + r[t+1] - δ)
        end

        
        F[T-1] = (K[T] - K̄) + λB * (B[T] - B̄)

        return
    end

    sol = nlsolve(
        residuals!, z0;
        method = :trust_region,
        xtol = 1e-10,
        ftol = 1e-10,
        iterations = 80_000,
        autoscale = false
    )

    @assert NLsolve.converged(sol) "PF solver did not converge. Try smaller shock_size, smaller ρa, or smaller φb."

    zstar = sol.zero
    I_dec = exp.(zstar)

    
    I = zeros(T)
    I[1:T-1] .= I_dec

    K = zeros(T); K[1] = K̄
    for t in 1:(T-1)
        K[t+1] = (1 - δ) * K[t] + I[t]
    end
    I[T] = δ * K[T]

    B = zeros(T); B[1] = B̄
    C = zeros(T); N = zeros(T); Y = zeros(T); τ = zeros(T); Ttax = zeros(T)
    r = zeros(T)

    for t in 1:T
        N[t] = labor_GHH(K[t], KG[t], A[t])
        Y[t] = prod(K[t], KG[t], N[t], A[t])
        r[t] = α * Y[t] / K[t]

        τ[t] = taxrate(B[t], Y[t])
        Ttax[t] = τ[t] * Y[t]

        
        C[t] = Y[t] - I[t] - IG[t] - i * B[t]

        if t < T
            B[t+1] = IG[t] + (1 + i) * B[t] - Ttax[t]
        end
    end

    pctlog_pos(x, x̄) = 100 .* log.(max.(x, ϵ) ./ max(x̄, ϵ))
    pctdev(x, x̄)     = 100 .* ((x ./ x̄) .- 1.0)

    return (
        t  = 0:(T-1),
        A  = pctlog_pos(A, Ā),
        Y  = pctlog_pos(Y, Ȳ),
        C  = pctlog_pos(C, C̄),
        I  = pctdev(I, Ī),
        N  = pctlog_pos(N, N̄),
        K  = pctlog_pos(K, K̄),
        KG = pctlog_pos(KG, KḠ),
        IG = pctlog_pos(IG, IḠ),
        B  = pctdev(B, B̄),
        τ  = 100 .* (τ .- τ̄)
    )
end

################################################################################
# RUN + PLOT
################################################################################
irfA_low  = irf_tfp_shock_pf_logI(param_low,  ss_low;  T=200, shock_size=0.01, ρa= 0.9490, φb=0.109277, φy=0.144772)
irfA_high = irf_tfp_shock_pf_logI(param_high, ss_high; T=200, shock_size=0.01, ρa= 0.9490, φb=0.104832, φy=0.064843)

function plot_irf_grid_low_high_tfp(irf_low, irf_high; main_title="IRF to TFP Shock")
    t = irf_low.t

    function twoline(y_low, y_high; ttl="", xlab="", ylab="% dev")
        p = plot(t, y_low,  lw=3, label="Low debt", title=ttl, xlabel=xlab, ylabel=ylab)
        plot!(p, t, y_high, lw=3, ls=:dash, label="High debt")
        hline!(p, [0.0], ls=:dot, label="")
        return p
    end

    p1 = hasproperty(irf_low, :A) ? twoline(irf_low.A,  irf_high.A,  ttl="TFP (A)") :
                                    twoline(irf_low.Y .* 0, irf_high.Y .* 0, ttl="TFP (A) (not returned)")

    p2 = twoline(irf_low.Y,  irf_high.Y,  ttl="Output (Y)")
    p3 = twoline(irf_low.C,  irf_high.C,  ttl="Consumption (C)")
    p4 = twoline(irf_low.I,  irf_high.I,  ttl="Private Investment (I)", xlab="Periods")
    p5 = twoline(irf_low.K,  irf_high.K,  ttl="Private Capital (K)",     xlab="Periods")
    p6 = twoline(irf_low.N,  irf_high.N,  ttl="Labor (N)",               xlab="Periods")

    plot(p1, p2, p3, p4, p5, p6,
         layout=(2,3),
         size=(1200,650),
         plot_title=main_title)
end

default(left_margin=10Plots.mm, bottom_margin=8Plots.mm, top_margin=6Plots.mm, right_margin=6Plots.mm)
display(plot_irf_grid_low_high_tfp(irfA_low, irfA_high; main_title="IRF to TFP Shock"))

######################################################################################################################
################################### Moment matching exercise
######################################################################################################################
const TFP_IRF_FUN = irf_tfp_shock_pf_logI
const IG_IRF_FUN = irf_public_investment_pf_logI
const HP_LAMBDA = 100.0
const VARS_COMPARE = ["Y","C","I","N","IG","W","Prod"]

################################################################################
#  SHOCK SMM 
################################################################################
logistic(x) = 1 / (1 + exp(-x))
logit(p)    = log(p/(1-p))

function detrend_linear(y::AbstractVector)
    yv = collect(Float64, y)
    T  = length(yv)
    t  = collect(1.0:T)
    X  = hcat(ones(T), t)
    β  = X \ yv
    return yv .- X * β
end

function acf1(y::AbstractVector)
    yv = collect(Float64, y)
    T  = length(yv)
    @assert T >= 3
    y0 = yv .- mean(yv)
    num = dot(y0[2:end], y0[1:end-1])
    den = dot(y0[1:end-1], y0[1:end-1])
    return num / den
end

function moments_for_matching(y::AbstractVector)
    yv = collect(Float64, y)
    return (std = std(yv), ac1 = acf1(yv))
end

function simulate_ar1(ρ::Float64, σ::Float64; T::Int=200, burn::Int=2000, seed::Int=123)
    @assert 0.0 < ρ < 1.0
    @assert σ > 0.0
    rng   = MersenneTwister(seed)
    total = T + burn
    y     = zeros(total)
    for t in 2:total
        y[t] = ρ * y[t-1] + σ * randn(rng)
    end
    return y[(burn+1):end]
end

function compute_log_tfp(Y, K, N, α)
    Yv = Vector{Float64}(Y)
    Kv = Vector{Float64}(K)
    Nv = Vector{Float64}(N)
    return log.(Yv) .- α .* log.(Kv) .- (1.0 - α) .* log.(Nv)
end

function build_empirical_series_str(Dataset::DataFrame; α::Float64=0.33, split_year::Int=2007)
    req = ["GDP", "Private_Capital", "Labour", "Public_Investment", "Year"]
    nm  = String.(names(Dataset))
    missing_cols = setdiff(req, nm)
    @assert isempty(missing_cols) "Dataset is missing columns: $(missing_cols). Found: $(nm)"

    df = copy(Dataset)

    if !("DebtRegime" in String.(names(df)))
        df[!, "DebtRegime"] = ifelse.(df[!, "Year"] .<= split_year, "low", "high")
    end

    df = dropmissing(df, vcat(req, ["DebtRegime"]))

    df[!, "LogTFP"] = compute_log_tfp(df[!, "GDP"], df[!, "Private_Capital"], df[!, "Labour"], α)
    df[!, "LogIG"]  = log.(Float64.(df[!, "Public_Investment"]))

    out = Dict{String, NamedTuple}()
    for regime in unique(df[!, "DebtRegime"])
        sdf = df[df[!, "DebtRegime"] .== regime, :]
        sdf = sort(copy(sdf), "Year")

        tfp_stat = detrend_linear(sdf[!, "LogTFP"])
        ig_stat  = sdf[!, "LogIG"] .- mean(sdf[!, "LogIG"])

        out[string(regime)] = (year = collect(Int, sdf[!, "Year"]), tfp = tfp_stat, ig = ig_stat)
    end
    return out
end

function smm_objective_ar1(x::Vector{Float64}, y_emp::Vector{Float64};
        Tsim::Int=400, burn::Int=2000, seed::Int=123, w_std::Float64=1.0, w_ac1::Float64=1.0)

    ρ = logistic(x[1])
    σ = exp(x[2])

    m_emp = moments_for_matching(y_emp)
    y_sim = simulate_ar1(ρ, σ; T=Tsim, burn=burn, seed=seed)
    m_sim = moments_for_matching(y_sim)

    d_std = (m_sim.std - m_emp.std) / max(m_emp.std, 1e-12)
    d_ac1 = (m_sim.ac1 - m_emp.ac1) / max(abs(m_emp.ac1), 1e-12)

    return w_std * d_std^2 + w_ac1 * d_ac1^2
end

function smm_estimate_ar1(y_emp::Vector{Float64};
        ρ0::Float64=0.90, σ0::Float64=0.01, Tsim::Int=400, burn::Int=2000, seed::Int=123)

    x0 = [logit(ρ0), log(σ0)]

    res = optimize(
        x -> smm_objective_ar1(x, y_emp; Tsim=Tsim, burn=burn, seed=seed),
        x0,
        NelderMead(),
        Optim.Options(iterations = 800)
    )

    xhat = Optim.minimizer(res)
    ρhat = logistic(xhat[1])
    σhat = exp(xhat[2])

    m_emp = moments_for_matching(y_emp)
    y_sim = simulate_ar1(ρhat, σhat; T=Tsim, burn=burn, seed=seed)
    m_sim = moments_for_matching(y_sim)

    return (ρ=ρhat, σ=σhat, opt=res, emp=m_emp, sim=m_sim, ysim=y_sim)
end

function run_smm_all_str(Dataset::DataFrame; α::Float64=0.33, Tsim::Int=400, burn::Int=2000)
    series = build_empirical_series_str(Dataset; α=α)

    results = DataFrame(
        DebtRegime = String[],
        Process    = String[],
        rho        = Float64[],
        sigma      = Float64[],
        emp_std    = Float64[],
        emp_ac1    = Float64[],
        sim_std    = Float64[],
        sim_ac1    = Float64[],
        objective  = Float64[]
    )

    for regime in sort(collect(keys(series)))
        est_tfp = smm_estimate_ar1(series[regime].tfp; ρ0=0.90, σ0=0.01, Tsim=Tsim, burn=burn, seed=111)
        push!(results, (regime, "TFP", est_tfp.ρ, est_tfp.σ,
                        est_tfp.emp.std, est_tfp.emp.ac1, est_tfp.sim.std, est_tfp.sim.ac1,
                        Optim.minimum(est_tfp.opt)))

        est_ig = smm_estimate_ar1(series[regime].ig; ρ0=0.90, σ0=0.01, Tsim=Tsim, burn=burn, seed=222)
        push!(results, (regime, "IG", est_ig.ρ, est_ig.σ,
                        est_ig.emp.std, est_ig.emp.ac1, est_ig.sim.std, est_ig.sim.ac1,
                        Optim.minimum(est_ig.opt)))
    end

    return results
end

################################################################################
# BUILD DATA DF 
################################################################################
function BuildDataDF_fromYourVars(Dataset::DataFrame; split_year::Int=2007)
    df = copy(Dataset)

    req = ["Year","GDP","Consumption","Private_Investment","Labour","Public_Investment","Wages"]
    missing_cols = setdiff(req, String.(names(df)))
    @assert isempty(missing_cols) "Dataset is missing columns: $(missing_cols). Available: $(String.(names(df)))"

    out = DataFrame()
    out[!, "Year"] = Int.(df[!, "Year"])
    out[!, "DebtRegime"] = ("DebtRegime" in String.(names(df))) ? String.(df[!, "DebtRegime"]) :
                           ifelse.(out[!, "Year"] .<= split_year, "low", "high")

    out[!, "Y"]  = Float64.(df[!, "GDP"])
    out[!, "C"]  = Float64.(df[!, "Consumption"])
    out[!, "I"]  = Float64.(df[!, "Private_Investment"])
    out[!, "N"]  = Float64.(df[!, "Labour"])
    out[!, "IG"] = Float64.(df[!, "Public_Investment"])
    out[!, "W"]  = Float64.(df[!, "Wages"])
    out[!, "Prod"] = out[!, "Y"] ./ out[!, "N"]

    return dropmissing(out, ["Year","DebtRegime","Y","C","I","N","IG","W","Prod"])
end

################################################################################
#  MODEL SIMULATION USING YOUR IRFS 
################################################################################
function GetShockParams(results_smm::DataFrame; regime::String, process::String)
    idx = findfirst((results_smm.DebtRegime .== regime) .& (results_smm.Process .== process))
    @assert !isnothing(idx) "Missing (regime=$regime, process=$process) in results_smm."
    return (rho = results_smm.rho[idx], sigma = results_smm.sigma[idx])
end

function convolve_irf(ψ::Vector{Float64}, ε::Vector{Float64})
    T = length(ε)
    H = length(ψ)
    x = zeros(T)
    for t in 1:T
        s = 0.0
        hmax = min(H, t)
        @inbounds for h in 1:hmax
            s += ψ[h] * ε[t - h + 1]
        end
        x[t] = s
    end
    return x
end

function BuildSSNamedTuple(ss_any)
    if ss_any isa NamedTuple
       
        needed = (:Y, :C, :I, :N, :IG)
        for k in needed
            @assert k in keys(ss_any) "Your ss NamedTuple is missing key $k. Keys found: $(keys(ss_any))"
        end
        return ss_any
    else
        error("ss_low/ss_high must be NamedTuple with keys :Y,:C,:I,:N,:IG. You provided: $(typeof(ss_any)).")
    end
end


function IRF_TFP(p::NamedTuple, ss::NamedTuple; T::Int, shock::Float64, rho::Float64)
    return TFP_IRF_FUN(p, ss; T=T, shock_size=shock, ρa=rho)
end

function IRF_IG(p::NamedTuple, ss::NamedTuple; T::Int, shock::Float64, rho::Float64)
    return IG_IRF_FUN(p, ss; T=T, shock_size=shock, ρg=rho)
end


function SimulateModelSeries_IRF(p::NamedTuple, ss_in; results_smm::DataFrame,
                                 regime::String,
                                 T::Int=3000, burn::Int=500, H::Int=80, seed::Int=1)

    ss = BuildSSNamedTuple(ss_in)

    tfp = GetShockParams(results_smm; regime=regime, process="TFP")
    ig  = GetShockParams(results_smm; regime=regime, process="IG")

    irfA = IRF_TFP(p, ss; T=H, shock=1.0, rho=tfp.rho)
    irfG = IRF_IG(p, ss; T=H, shock=1.0, rho=ig.rho)

    
    needed = (:Y, :C, :I, :N, :IG)
    for v in needed
        @assert v in propertynames(irfA) "TFP IRF missing field $v. Found: $(propertynames(irfA))"
        @assert v in propertynames(irfG) "IG IRF missing field $v. Found: $(propertynames(irfG))"
    end

    rng  = MersenneTwister(seed)
    εA   = tfp.sigma .* randn(rng, T + burn)
    εG   = ig.sigma  .* randn(rng, T + burn)

    
    Yd  = convolve_irf(Float64.(irfA.Y),  εA) .+ convolve_irf(Float64.(irfG.Y),  εG)
    Cd  = convolve_irf(Float64.(irfA.C),  εA) .+ convolve_irf(Float64.(irfG.C),  εG)
    Id  = convolve_irf(Float64.(irfA.I),  εA) .+ convolve_irf(Float64.(irfG.I),  εG)
    Nd  = convolve_irf(Float64.(irfA.N),  εA) .+ convolve_irf(Float64.(irfG.N),  εG)
    IGd = convolve_irf(Float64.(irfA.IG), εA) .+ convolve_irf(Float64.(irfG.IG), εG)

   
    Yd  = Yd[(burn+1):end]
    Cd  = Cd[(burn+1):end]
    Id  = Id[(burn+1):end]
    Nd  = Nd[(burn+1):end]
    IGd = IGd[(burn+1):end]

    
    Y  = ss.Y  .* (1 .+ Yd)
    C  = ss.C  .* (1 .+ Cd)
    I  = ss.I  .* (1 .+ Id)
    N  = ss.N  .* (1 .+ Nd)
    IG = ss.IG .* (1 .+ IGd)

    ModelDF = DataFrame()
    ModelDF[!, "Y"]  = Y
    ModelDF[!, "C"]  = C
    ModelDF[!, "I"]  = I
    ModelDF[!, "N"]  = N
    ModelDF[!, "IG"] = IG
    ModelDF[!, "Prod"] = Y ./ N

   
    if (:W in propertynames(irfA)) && (:W in propertynames(irfG)) && (haskey(ss, :W))
        Wd = convolve_irf(Float64.(irfA.W), εA) .+ convolve_irf(Float64.(irfG.W), εG)
        Wd = Wd[(burn+1):end]
        ModelDF[!, "W"] = ss.W .* (1 .+ Wd)
    else
       
        ModelDF[!, "W"] = fill(NaN, size(ModelDF, 1))
    end

    return ModelDF
end

################################################################################
#  MOMENTS + OVERLEAF TABLES (STRING-ONLY)
################################################################################
function HPFilter(y::AbstractVector{<:Real}; λ::Float64=100.0)
    yv = collect(Float64, y)
    T  = length(yv)
    @assert T ≥ 6 "HP filter requires T ≥ 6."

    D = zeros(T-2, T)
    for i in 1:(T-2)
        D[i, i]   = 1.0
        D[i, i+1] = -2.0
        D[i, i+2] = 1.0
    end

    A = I(T) + λ * (D' * D)
    trend = A \ yv
    cycle = yv .- trend
    return trend, cycle
end

StdSafe(x) = std(collect(Float64, x))
function CorrSafe(x, y)
    xv = collect(Float64, x); yv = collect(Float64, y)
    sx = std(xv); sy = std(yv)
    (sx < 1e-14 || sy < 1e-14) && return NaN
    return cor(xv, yv)
end

function BuildCycles_str(df::DataFrame; λ::Float64=100.0,
                         Vars::Vector{String},
                         LogVars::Set{String}=Set(["Y","C","I","IG","Prod"]),
                         log_model::Bool=false)
    sdf = dropmissing(select(df, Vars))
    out = Dict{String, Vector{Float64}}()

    for v in Vars
        x = Float64.(sdf[!, v])

        
        if log_model == false && (v in LogVars)
            @assert all(x .> 0.0) "Variable $v must be positive to take logs."
            x = log.(x)
        end

        _, cyc = HPFilter(x; λ=λ)
        out[v] = cyc
    end

    return out
end


function MomentsTable_str(cycles::Dict{String,Vector{Float64}}; Vars::Vector{String})
    Yc = cycles["Y"]
    sY = StdSafe(Yc)

    df = DataFrame(Variable=String[], VolRelY=Float64[], CorrWithY=Float64[])
    for v in Vars
        xc = cycles[v]
        push!(df, (v, StdSafe(xc)/max(sY,1e-12), CorrSafe(xc, Yc)))
    end
    return df
end

function LatexTableVolatility_str(dfData::DataFrame, dfModel::DataFrame; VarMap=Dict{String,String}())
    getval(df, var::String, col::Symbol) = (idx = findfirst(df.Variable .== var); isnothing(idx) ? NaN : df[idx,col])
    order = vcat(["Y"], filter(!=("Y"), dfData.Variable))

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Model vs.\\ Data: Volatilities Relative to Output}")
    push!(lines, "\\label{tab:volatilities}")
    push!(lines, "\\begin{tabular}{lcc}")
    push!(lines, "\\toprule")
    push!(lines, "Variable & Data & Model \\\\")
    push!(lines, "\\midrule")
    for v in order
        name = get(VarMap, v, v)
        d = getval(dfData,  v, :VolRelY)
        m = getval(dfModel, v, :VolRelY)
        push!(lines, @sprintf("%s & %.2f & %.2f \\\\", name, d, m))
    end
    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")
    return join(lines, "\n")
end

function LatexTableCorrelation_str(dfData::DataFrame, dfModel::DataFrame; VarMap=Dict{String,String}())
    getval(df, var::String, col::Symbol) = (idx = findfirst(df.Variable .== var); isnothing(idx) ? NaN : df[idx,col])
    order = filter(!=("Y"), dfData.Variable)

    lines = String[]
    push!(lines, "\\begin{table}[H]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Model vs.\\ Data: Correlations with Output}")
    push!(lines, "\\label{tab:correlations}")
    push!(lines, "\\begin{tabular}{lcc}")
    push!(lines, "\\toprule")
    push!(lines, "Variable & Data & Model \\\\")
    push!(lines, "\\midrule")
    for v in order
        name = get(VarMap, v, v)
        d = getval(dfData,  v, :CorrWithY)
        m = getval(dfModel, v, :CorrWithY)
        push!(lines, @sprintf("%s & %.2f & %.2f \\\\", name, d, m))
    end
    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")
    return join(lines, "\n")
end

function BuildModelVsDataTables_str(DataDF::DataFrame, ModelDF::DataFrame;
                                    λ::Float64=HP_LAMBDA,
                                    Vars::Vector{String}=VARS_COMPARE)
    haveD = Set(String.(names(DataDF)))
    haveM = Set(String.(names(ModelDF)))
    VarsUse = [v for v in Vars if (v in haveD) && (v in haveM)]
    @assert "Y" in VarsUse "Need column \"Y\" in both DataDF and ModelDF."

    CycData  = BuildCycles_str(DataDF;  λ=λ, Vars=VarsUse, log_model=false)
    CycModel = BuildCycles_str(ModelDF; λ=λ, Vars=VarsUse, log_model=true)

    MomData  = MomentsTable_str(CycData;  Vars=VarsUse)
    MomModel = MomentsTable_str(CycModel; Vars=VarsUse)

    VarMap = Dict(
        "Y"    => "Output (Y)",
        "C"    => "Consumption (C)",
        "I"    => "Private Investment (I)",
        "N"    => "Labour (N)",
        "IG"   => "Public Investment (I^G)",
        "W"    => "Real Wage (w)",
        "Prod" => "Labour Productivity (Y/N)"
    )

    TexVol  = LatexTableVolatility_str(MomData, MomModel; VarMap=VarMap)
    TexCorr = LatexTableCorrelation_str(MomData, MomModel; VarMap=VarMap)

    return (MomData=MomData, MomModel=MomModel, TexVol=TexVol, TexCorr=TexCorr)
end

################################################################################
#  RUN PIPELINE
################################################################################
# estimate shocks
results_smm = run_smm_all_str(Dataset; α=0.33, Tsim=400, burn=2000)
println(results_smm)

# build data DF and split regimes
DataDF = BuildDataDF_fromYourVars(Dataset; split_year=2007)
DataLow  = DataDF[DataDF[!, "DebtRegime"] .== "low",  Not("DebtRegime")]
DataHigh = DataDF[DataDF[!, "DebtRegime"] .== "high", Not("DebtRegime")]

# simulate model 
Tsim_series = 3000
burn_series = 500
H           = 80

ssL = BuildSSNamedTuple(ss_low)
ssH = BuildSSNamedTuple(ss_high)

ModelLow  = SimulateModelSeries_IRF(param_low,  ssL; results_smm=results_smm, regime="low",
                                   T=Tsim_series, burn=burn_series, H=H, seed=11)

ModelHigh = SimulateModelSeries_IRF(param_high, ssH; results_smm=results_smm, regime="high",
                                   T=Tsim_series, burn=burn_series, H=H, seed=22)

# Output tables
OutLow  = BuildModelVsDataTables_str(DataLow,  ModelLow;  λ=HP_LAMBDA, Vars=VARS_COMPARE)
OutHigh = BuildModelVsDataTables_str(DataHigh, ModelHigh; λ=HP_LAMBDA, Vars=VARS_COMPARE)

println("\n--- LOW DEBT: Volatilities ---\n")
println(OutLow.TexVol)
println("\n--- LOW DEBT: Correlations ---\n")
println(OutLow.TexCorr)

println("\n--- HIGH DEBT: Volatilities ---\n")
println(OutHigh.TexVol)
println("\n--- HIGH DEBT: Correlations ---\n")
println(OutHigh.TexCorr)





