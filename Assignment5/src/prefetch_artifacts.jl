using Pkg, Artifacts, Base.BinaryPlatforms, Printf

const TARGET = Platform("aarch64", "linux"; libc = "glibc", cxxstring_abi = "cxx11",
                        julia_version = v"1.12.6", cuda = get(ENV, "PREFETCH_CUDA", "13.0"),
                        tegra = "none", llvm_version = "18")
const DRY_RUN = get(ENV, "PREFETCH_DRY_RUN", "0") == "1"

Pkg.activate(@__DIR__)
println("target platform: ", triplet(TARGET), DRY_RUN ? " (dry run)" : "")

todo = Tuple{String,String,Dict{String,Any},String}[]
for (uuid, info) in Pkg.dependencies()
    info.source === nothing && continue
    toml = joinpath(info.source, "Artifacts.toml")
    isfile(toml) || continue
    for (name, meta) in select_downloadable_artifacts(toml; platform = TARGET, include_lazy = true)
        push!(todo, (info.name, name, meta, toml))
    end
end
missing_sources = [info.name for (_, info) in Pkg.dependencies() if info.source !== nothing && !isdir(info.source)]
isempty(missing_sources) || println("package sources not downloaded yet: ", join(missing_sources, ", "))

t0 = time()
for (pkg, name, meta, toml) in sort(todo; by = first)
    hash = Base.SHA1(meta["git-tree-sha1"])
    have = artifact_exists(hash)
    tags = join(["$k=$v" for (k, v) in meta if k in ("cuda", "cuda_platform", "tegra", "llvm_version", "lazy")], " ")
    @printf("%-22s %-24s %-8s %s\n", pkg, name, have ? "present" : "missing", tags)
    (DRY_RUN || have) && continue
    t = time()
    Pkg.Artifacts.ensure_artifact_installed(name, meta, toml; platform = TARGET)
    @printf("    downloaded in %.0f s\n", time() - t)
end
@printf("done in %.0f s | %d artifacts selected for %s\n", time() - t0, length(todo), triplet(TARGET))
