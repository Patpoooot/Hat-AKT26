using CUDA, LinearAlgebra, Printf, Random

const TEAM_ID     = 1013
const SEED        = 2000 + TEAM_ID
const MODE        = get(ENV, "CHOL_MODE", "explore")
intlist(s) = Tuple(parse.(Int, split(s, r"[:,; ]+")))
const SIZES       = haskey(ENV, "CHOL_SIZES") ? intlist(ENV["CHOL_SIZES"]) :
                    MODE == "quick" ? (1000:1000:3000) : (1000:500:7000)
const NBS         = haskey(ENV, "CHOL_NBS") ? intlist(ENV["CHOL_NBS"]) :
                    MODE == "quick" ? (256, 512) : (256, 384, 512, 768, 1024, 1536, 2048)
const REPS        = parse(Int, get(ENV, "CHOL_REPS", "3"))
const TIME_BUDGET = parse(Float64, get(ENV, "CHOL_TIME_BUDGET", "230"))
const T_START     = time()
const RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results"))
const CSV_FOR     = Dict("quick"   => joinpath(RESULTS_DIR, "exploration", "results_gpu_team$(TEAM_ID)_quick.csv"),
                        "explore" => joinpath(RESULTS_DIR, "exploration", "results_gpu_team$(TEAM_ID)_explore.csv"),
                        "final"   => joinpath(RESULTS_DIR, "results_gpu_team$(TEAM_ID).csv"))

tile(i::Int, nb::Int, n::Int) = ((i - 1) * nb + 1):min(i * nb, n)
chol_flops(n) = n^3 / 3 + n^2 / 2 + n / 6
gflops(flops, t) = flops / t / 1e9

function generate_spd(n::Integer; T::Type{<:AbstractFloat} = Float64, seed::Integer = SEED)
    M = rand(CUDA.RNG(seed), T, n, n)
    A = M * M'
    view(A, diagind(A)) .+= T(n)
    return (A + permutedims(A)) ./ 2
end

generate_solution(n::Integer; T::Type{<:AbstractFloat} = Float64, seed::Integer = SEED + 1) =
    rand(CUDA.RNG(seed), T, n) .+ one(T)

function cholesky_tiled_gpu!(A::CuMatrix{T}, nb::Int; variant::Symbol = :panel) where {T<:AbstractFloat}
    n = size(A, 1); p = cld(n, nb)
    for k in 1:p
        kk  = tile(k, nb, n)
        Akk = view(A, kk, kk)
        _, info = CUSOLVER.potrf!('L', Akk)
        info > 0 && throw(PosDefException(first(kk) + info - 1))
        k == p && break
        if variant == :tile
            for i in k+1:p
                CUBLAS.trsm!('R', 'L', 'T', 'N', one(T), Akk, view(A, tile(i, nb, n), kk))
            end
            for j in k+1:p, i in j:p
                Aik = view(A, tile(i, nb, n), kk)
                if i == j
                    CUBLAS.syrk!('L', 'N', -one(T), Aik, one(T), view(A, tile(i, nb, n), tile(i, nb, n)))
                else
                    Ajk = view(A, tile(j, nb, n), kk)
                    CUBLAS.gemm!('N', 'T', -one(T), Aik, Ajk, one(T), view(A, tile(i, nb, n), tile(j, nb, n)))
                end
            end
        else
            rest = last(kk)+1:n
            Ark  = view(A, rest, kk)
            CUBLAS.trsm!('R', 'L', 'T', 'N', one(T), Akk, Ark)
            CUBLAS.syrk!('L', 'N', -one(T), Ark, one(T), view(A, rest, rest))
        end
    end
    return A
end

function cholesky_cusolver!(A::CuMatrix)
    _, info = CUSOLVER.potrf!('L', A)
    info > 0 && throw(PosDefException(info))
    return A
end

function chol_solve!(F::CuMatrix{T}, x::CuVector{T}) where {T}
    CUBLAS.trsv!('L', 'N', 'N', F, x)
    CUBLAS.trsv!('L', 'T', 'N', F, x)
    return x
end

function backward_error(A::CuMatrix, x::CuVector, b::CuVector)
    A64, x64, b64 = Float64.(A), Float64.(x), Float64.(b)
    r = b64 - A64 * x64
    normA = maximum(sum(abs, A64; dims = 2))
    return maximum(abs, r) / (normA * maximum(abs, x64) + maximum(abs, b64))
end

function bench_gpu(f!, A0::CuMatrix; reps::Int = REPS)
    A = copy(A0)
    f!(A)
    best = Inf
    for _ in 1:reps
        copyto!(A, A0)
        best = min(best, Float64(CUDA.@elapsed f!(A)))
    end
    return best
end

const PEAKS_GFLOPS = [
    "H100 PCIe" => (fp32 = 51_200.0, fp64 = 51_200.0, tf32 = 378_000.0),
    "H100 NVL"  => (fp32 = 60_000.0, fp64 = 60_000.0, tf32 = 417_500.0),
    "H100"      => (fp32 = 66_900.0, fp64 = 66_900.0, tf32 = 494_700.0),
    "H200"      => (fp32 = 66_900.0, fp64 = 66_900.0, tf32 = 494_700.0),
    "A100"      => (fp32 = 19_500.0, fp64 = 19_500.0, tf32 = 156_000.0),
    "L40S"      => (fp32 = 91_600.0, fp64 = 1_431.0,  tf32 = 183_000.0),
    "V100"      => (fp32 = 15_700.0, fp64 = 7_800.0,  tf32 = NaN),
]

nvml_device(dev = device()) = CUDA.NVML.Device(CUDA.uuid(dev))

function computed_peaks(dev = device())
    cc  = capability(dev)
    sms = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    ghz = try
        CUDA.NVML.max_clock_info(nvml_device(dev)).sm / 1e3
    catch
        attribute(dev, CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE) / 1e6
    end
    fp32 = sms * (cc >= v"8.6" ? 128 : 64) * 2 * ghz
    fp64 = fp32 * (cc in (v"6.0", v"7.0", v"8.0", v"9.0", v"10.0") ? 1 / 2 : 1 / 64)
    return (fp32 = fp32, fp64 = fp64, tf32 = NaN)
end

function gpu_peaks(name::AbstractString = CUDA.name(device()))
    for (key, peaks) in PEAKS_GFLOPS
        occursin(key, name) && return peaks
    end
    return computed_peaks()
end

peak_for(T::Type{<:AbstractFloat}, peaks = gpu_peaks()) = T == Float32 ? peaks.fp32 : peaks.fp64

gpu_energy_j() = try
    Float64(CUDA.NVML.energy_consumption(nvml_device()))
catch
    NaN
end

function measure(impl::AbstractString, f!, A0::CuMatrix{T}, b::CuVector{T}; nb::Int = 0, reps::Int = REPS) where {T}
    n = size(A0, 1)
    e0, w0 = gpu_energy_j(), time()
    t = bench_gpu(f!, A0; reps = reps)
    x = chol_solve!(f!(copy(A0)), copy(b))
    η = backward_error(A0, x, b)
    CUDA.synchronize()
    return (impl = String(impl), precision = string(T), N = n, nb = nb, time_s = t,
            gflops = gflops(chol_flops(n), t), backward_error = η, passed = η <= 10 * n * eps(T),
            gpu = CUDA.name(device()), peak_gflops = peak_for(T),
            wall_s = time() - w0, energy_j = gpu_energy_j() - e0)
end

function warmup(T::Type{<:AbstractFloat}; seconds::Real = 1.0)
    A = generate_spd(2000; T = T)
    b = A * generate_solution(2000; T = T)
    for variant in (:tile, :panel)
        cholesky_tiled_gpu!(copy(A), 512; variant = variant)
    end
    F = cholesky_cusolver!(copy(A))
    t0 = time()
    while time() - t0 < seconds
        copyto!(F, A)
        cholesky_cusolver!(F)
        CUDA.synchronize()
    end
    backward_error(A, chol_solve!(F, copy(b)), b)
    return nothing
end

function append_csv(path::AbstractString, rows)
    isempty(rows) && return path
    mkpath(dirname(path))
    new_file = !isfile(path)
    open(path, "a") do io
        new_file && println(io, join(keys(first(rows)), ","))
        for r in rows
            println(io, join(values(r), ","))
        end
    end
    return path
end

function read_csv(path::AbstractString)
    lines  = filter(!isempty, readlines(path))
    header = Tuple(Symbol.(split(lines[1], ",")))
    parse_field(s) = something(tryparse(Int, s), tryparse(Float64, s), tryparse(Bool, s), String(s))
    return [NamedTuple{header}(Tuple(parse_field.(split(l, ",")))) for l in lines[2:end]]
end

function best_configs(rows)
    best = Dict{Tuple{String,Int},Any}()
    for r in rows
        startswith(r.impl, "tiled_") || continue
        key = (r.precision, r.N)
        if !haskey(best, key) || r.gflops > best[key].gflops
            best[key] = r
        end
    end
    return Dict(key => [(Symbol(chopprefix(r.impl, "tiled_")), r.nb)] for (key, r) in best)
end

function gpu_summary()
    dev = device()
    return @sprintf("%s | compute capability %s | %d SMs | %.1f GiB | driver %s | CUDA runtime %s",
                    CUDA.name(dev), capability(dev),
                    attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
                    CUDA.totalmem(dev) / 2^30, CUDA.driver_version(), CUDA.runtime_version())
end

function run_benchmarks(; sizes = SIZES, precisions = (Float32, Float64), nbs = NBS,
                        variants = (:tile, :panel), configs = nothing, reps::Int = REPS,
                        csv::AbstractString = CSV_FOR["explore"], deadline::Real = Inf)
    println(gpu_summary())
    isfile(csv) && rm(csv)
    foreach(warmup, precisions)
    e0, t0 = gpu_energy_j(), time()
    rows = NamedTuple[]
    for n in sizes, T in precisions
        if time() > deadline
            @warn "time budget reached, stopping before N = $n ($T)"
            break
        end
        A0 = generate_spd(n; T = T)
        b  = A0 * generate_solution(n; T = T)
        trials = configs === nothing ? [(v, nb) for v in variants for nb in nbs if nb < n] :
                                    get(configs, (string(T), n), Tuple{Symbol,Int}[])
        block = NamedTuple[]
        for (variant, nb) in trials
            push!(block, measure("tiled_$(variant)", A -> cholesky_tiled_gpu!(A, nb; variant = variant),
                                A0, b; nb = nb, reps = reps))
        end
        push!(block, measure("cusolver", cholesky_cusolver!, A0, b; reps = reps))
        append_csv(csv, block)
        append!(rows, block)
        ours = filter(r -> startswith(r.impl, "tiled"), block)
        best_text = if isempty(ours)
            "no tiled configuration"
        else
            best = ours[argmax([r.gflops for r in ours])]
            @sprintf("best tiled %9.1f GFlop/s (%s, nb = %4d)", best.gflops, best.impl, best.nb)
        end
        @printf("%-7s N = %4d | %s | cusolver %9.1f GFlop/s | %5.1f J | all passed: %s\n",
                string(T), n, best_text, last(block).gflops,
                sum(r.energy_j for r in block), all(r -> r.passed, block))
        A0 = nothing; b = nothing
        GC.gc(false); CUDA.reclaim()
    end
    @printf("benchmark: %.1f s, GPU energy %.0f J, wrote %s\n", time() - t0, gpu_energy_j() - e0, csv)
    return rows
end

function main()
    haskey(CSV_FOR, MODE) || error("CHOL_MODE must be quick, explore or final (got $MODE)")
    println("mode = $MODE | reps = $REPS | time budget = $(TIME_BUDGET) s")
    MODE == "quick" && CUDA.versioninfo()
    configs = if MODE == "final"
        explore_rows = reduce(vcat, read_csv.(split(get(ENV, "CHOL_EXPLORE_CSV", CSV_FOR["explore"]), ":")))
        best_configs(explore_rows)
    else
        nothing
    end
    run_benchmarks(; configs = configs, csv = get(ENV, "CHOL_CSV", CSV_FOR[MODE]), deadline = T_START + TIME_BUDGET)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
