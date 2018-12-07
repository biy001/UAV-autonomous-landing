
include("UAVLanding.jl")
include("visual.jl")

POMCPOW_solver = POMCPOWSolver(max_depth = 1000, # the deeper, the more oscillations on heading angles.
                     tree_queries=1000,

                     criterion=MaxUCB(10.0),
                     enable_action_pw=true,
                     check_repeat_obs=false,
                     rng=MersenneTwister(2))

my_pomdp = UAVchasePOMDP()
policyPOMCPOW = POMCPOWPlanner(POMCPOW_solver, my_pomdp)
hr = HistoryRecorder(show_progress=true, max_steps=400)
up = SIRParticleFilter(my_pomdp, 400)
hist = simulate(hr, my_pomdp, policyPOMCPOW, up)

# view the simulation history:
# for (s, b, a, r, sp, o) in view(hist, 145:145)
#     @show s, a, r, o
# end
print(" Number of steps: ")
print(length(hist))

# plot a static plot showing the UAV and target trace:
# plt_static = plot3d(my_pomdp, hist)

# create a gif
gr()
frames = Frames(MIME("image/png"), fps=5)
@showprogress "Creating gif..." for i in 1:30
    push!(frames, plot3d(my_pomdp, view(hist, 1:i)))
end
write("out.gif", frames)
