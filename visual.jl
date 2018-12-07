
@recipe function f(pomdp::UAVchasePOMDP, h::AbstractPOMDPHistory{MDPState})
    mdp = pomdp.mdp
    ratio --> :equal
    # xlabel := "x (m)"
    # ylabel := "y (m)"
    # zlabel := "z (m)"
    xlim --> (-1, 12)
    ylim --> (-1, 10)
    zlim --> (0, 5)
    @series begin
        seriestype := :path
        label := "UAV path"
        linewidth := 2
        x = [s.uavPose[1] for s in state_hist(h)[1:end-1]]
        y = [s.uavPose[2] for s in state_hist(h)[1:end-1]]
        z = [s.uavPose[3] for s in state_hist(h)[1:end-1]]
        # color --> :orange
        color := :orange
        x, y, z
    end
    @series begin
        seriestype := :scatter
        label := "current UAV"
        pos = state_hist(h)[end-1].uavPose
        color := :orange
        [pos[1]], [pos[2]], [pos[3]]
    end
    @series begin
        label := "belief"
        pomdp, belief_hist(h)[end]
    end
    @series begin
        seriestype := :path
        label := "target path"
        linewidth := 2
        x = [s.targetPose[1] for s in state_hist(h)[1:end-1]]
        y = [s.targetPose[2] for s in state_hist(h)[1:end-1]]
        z = [mdp.target_height for s in state_hist(h)[1:end-1]]
        # color --> :green
        color := :green
        x, y, z
    end
    @series begin
        seriestype := :path
        label := "true target path"
        linewidth := 2
        x = [s.targetPose_true[1] for s in state_hist(h)[1:end-1]]
        y = [s.targetPose_true[2] for s in state_hist(h)[1:end-1]]
        z = [mdp.target_height for s in state_hist(h)[1:end-1]]
        # color --> :blue
        color := :blue
        x, y, z
    end
    @series begin
        if length(h) == 1
            color := :red
        elseif observation_hist(h)[end-1].if_observed
            color := :green
        else
            color := :red
        end
        seriestype := :scatter
        label := "estimated target"
        pos = state_hist(h)[end-1].targetPose
        [pos[1]], [pos[2]], [mdp.target_height]
    end
    @series begin
        color := :blue
        seriestype := :scatter
        label := "true target"
        pos = state_hist(h)[end-1].targetPose_true
        [pos[1]], [pos[2]], [mdp.target_height]
    end
end

@recipe function f(pomdp::UAVchasePOMDP, pc::ParticleCollection{MDPState})
    mdp = pomdp.mdp
    seriestype := :scatter
    x = [p.targetPose[1] for p in particles(pc)]
    y = [p.targetPose[2] for p in particles(pc)]
    z = [mdp.target_height for i in 1:n_particles(pc)]
    markersize --> [10.0*sqrt(weight(pc,i)) for i in 1:n_particles(pc)]
    color --> :black
    # markersize --> 0.1
    x, y, z
end
# not for visual but for particle filter
# @recipe function f(s::MDPState)
#     [s.targetPose[1]], [s.targetPose[2]]
# end
