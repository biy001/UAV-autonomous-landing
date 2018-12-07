

"""
Problem Overview:
A UAV chases a moving ground platform (the target) and intends to finally land on it.
Assume full knowledge of the UAV state with the only uncertainty lying on the target state.
A camera installed on the UAV receives intermittent observations of the target to estimate its state.
Assume a noisy measurement of the target location (2D) when it is inside the camera field-of-view,
and no measurement when it is outside.

"""

using POMDPs
using Random # for AbstractRNG
using POMDPModelTools # for Deterministic
using Parameters
using StaticArrays
using Distributions
using LinearAlgebra
using Base
using ParticleFilters
using POMCPOW
using MCTS
using Test
using POMDPSimulators
using POMDPPolicies
using Plots
# using PyPlot
using Reel
using ProgressMeter
# importall POMDPs

const Vec2 = SVector{2, Float64}
const Vec3 = SVector{3, Float64}

struct MDPState
    uavPose::Vec3
    uavHeading::Float64 # degree
    targetPose::Vec2
    targetPose_true::Vec2
end

struct MDPAction
    xy_speed::Float64 # m/s
    z_speed::Float64
    angle::Float64 # degree
end

struct POMDPAction
    look::Bool
    vel_steer::MDPAction
end


mutable struct targetObservationDistribution # not an actual distribution. Probably better to revise the name
    current_observation::Vec2
    if_observed::Bool
end
POMDPs.rand(rng::AbstractRNG, d::targetObservationDistribution) = d
# POMDPs.pdf(d::TMazeObservationDistribution, o::Int64) = o == d.current_observation ? (return 1.0) : (return 0.0)


@with_kw struct UAVchaseMDP <: MDP{MDPState, MDPAction}
    # mu::Float64          = 2.0
    target_velocity::Vec2 = SVector(0.2, 0.0)
    target_std::Float64     = 0.04# already very good: 0.01

    dt::Float64          = 0.1
    target_height::Float64 = 0.3 # height of the platform
    landing_radius::Float64  = 0.2 # old is 0.8
    ready_to_land_distance = 5.0 # ready for landing when target is this distance away from the origin
    UAV_xy_speed = 0.6
    UAV_z_speed = 0.2
    # initial conditions:
    init_UAVPose = SVector(0,0,0.5)
    init_UAVHeading = 0.0
    init_targetPose = SVector(3., -1.0)
    init_targetPose_true = SVector(2., 2.0)
    init_target_std = 5.
    # physical constraints:
    z_min::Float64 = 0.3
    z_max::Float64 = 5
    # rewards:
    r_undetected::Float64 = -800.
    r_outScene::Float64 = -100.0 # under the ground or above the ceiling
    r_action::Float64 = -10.0
    r_distance::Float64 = -300.

    discount::Float64    = 0.95
end


@with_kw struct UAVchasePOMDP <: POMDP{MDPState, MDPAction, targetObservationDistribution}
    mdp::UAVchaseMDP           = UAVchaseMDP()
    # meas_std::Float64          = 0.001 # camera measurement noise
    meas_std::Float64          = 0.1

    # angle to be within Field of View (FOV)
    FOV_angle1::Float64 = 40 # in degree
    FOV_angle2::Float64 = 40 # in degree
end

const UAVchaseProblem = Union{UAVchasePOMDP, UAVchaseMDP}
mdp(p::UAVchaseMDP) = p
mdp(p::UAVchasePOMDP) = p.mdp

POMDPs.discount(pp::UAVchasePOMDP) = mdp(pp).discount

function POMDPs.isterminal(pp::UAVchasePOMDP, s::MDPState)
    p = mdp(pp)
    target3Dpose = SVector(s.targetPose_true[1], s.targetPose_true[2], p.target_height)
    distance = norm(s.uavPose - target3Dpose)
    return (norm(target3Dpose) > p.ready_to_land_distance) && (distance < p.landing_radius)
end

mutable struct MDPStateDistribution
    current_uavPose::Vec3
    current_uavHeading::Float64
    current_targetPose_true::Vec2
    mean_targetPose::Vec2
    std_targetPose::Float64
end
POMDPs.sampletype(::Type{MDPStateDistribution}) = MDPState

function POMDPs.rand(rng::AbstractRNG, d::MDPStateDistribution)
    target_cov = Matrix(d.std_targetPose*Diagonal{Float64}(I, 2))
    rand_targetPose = d.mean_targetPose + rand(rng, MvNormal(target_cov))
    return MDPState(d.current_uavPose, d.current_uavHeading, rand_targetPose,  d.current_targetPose_true)
end

function POMDPs.initialstate_distribution(pp::UAVchaseProblem)
    p = mdp(pp)
    return MDPStateDistribution(p.init_UAVPose, p.init_UAVHeading, p.init_targetPose_true, p.init_targetPose, p.init_target_std)
end

function POMDPs.generate_s(pp::UAVchaseProblem, s::MDPState, a::MDPAction, rng::AbstractRNG)
    p = mdp(pp)
    # calculate target state
    target_dt_distance = p.dt*p.target_velocity
    ## sensor_noise = Base.rand(Normal(0, p.target_std), 2)
    target_dynamic_cov = Matrix(p.target_std*Diagonal{Float64}(I, 2))
    sensor_noise = rand(rng, MvNormal(target_dynamic_cov))
    ## sensor_noise = SVector(0,0)
    curr_targ_true = s.targetPose_true + target_dt_distance
    if if_in_FOV(pp, s)
        curr_targ = curr_targ_true + sensor_noise
    else
        curr_targ = s.targetPose + target_dt_distance + sensor_noise # next_target_pos(p, s.targetPose)
    end
    # calculate UAV state
    curr_angle = s.uavHeading + a.angle
    xy_dt_distance = p.dt*a.xy_speed*SVector(cosd(curr_angle), sind(curr_angle)) # careful
    z_dt_distance = p.dt*a.z_speed
    xyz_dt_distance = SVector(xy_dt_distance[1], xy_dt_distance[2], z_dt_distance)
    curr_pos = s.uavPose + xyz_dt_distance
    return MDPState(curr_pos, curr_angle, curr_targ, curr_targ_true)
end

function if_in_FOV(p::UAVchasePOMDP, s::MDPState)
    # pose = s.uavPose
    range1 = s.uavPose[3]*tand(p.FOV_angle1)
    range2 = s.uavPose[3]*tand(p.FOV_angle2)
    # within_x = (s.targetPose[1]>(s.uavPose[1]-range1))||(s.targetPose[1]<(s.uavPose[1]+range1))
    if !((s.targetPose_true[1]>(s.uavPose[1]-range1))&&(s.targetPose_true[1]<(s.uavPose[1]+range1)))
        return false # if target is NOT within FOV in x direction
    end
    if (s.targetPose_true[2]>(s.uavPose[2]-range2))&&(s.targetPose_true[2]<(s.uavPose[2]+range2))
        return true # if target is within FOV in y direction
    else
        return false
    end
end

function if_in_FOV_fake(p::UAVchasePOMDP, s::MDPState)
    # pose = s.uavPose
    range1 = s.uavPose[3]*tand(p.FOV_angle1)
    range2 = s.uavPose[3]*tand(p.FOV_angle2)
    # within_x = (s.targetPose[1]>(s.uavPose[1]-range1))||(s.targetPose[1]<(s.uavPose[1]+range1))
    if !((s.targetPose[1]>(s.uavPose[1]-range1))&&(s.targetPose[1]<(s.uavPose[1]+range1)))
        return false # if target is NOT within FOV in x direction
    end
    if (s.targetPose[2]>(s.uavPose[2]-range2))&&(s.targetPose[2]<(s.uavPose[2]+range2))
        return true # if target is within FOV in y direction
    else
        return false
    end
end
# I only count the case of POMDP
function POMDPs.reward(pp::UAVchasePOMDP, a::MDPAction, sp::MDPState)
    p = mdp(pp)
    # action_reward = p.r_action * ((a.xy_speed != 0)||(a.z_speed != 0))
    if ((a.xy_speed != 0)||(a.z_speed != 0))
        action_reward = p.r_action
    else
        action_reward = 0.0
    end
    # detection_reward = p.r_detect * if_in_FOV(pp, sp)
    if if_in_FOV(pp, sp)
        detection_reward = 0.0
    else
        detection_reward = p.r_undetected
    end
    # outScene_reward = p.r_outScene * ((sp.uavPose[3] < p.z_min)||(sp.uavPose[3] > p.z_max))
    if ((sp.uavPose[3] < p.z_min)||(sp.uavPose[3] > p.z_max))
        outScene_reward = p.r_outScene
    else
        outScene_reward = 0.0
    end
    # distance reward:
    target3Dpose = SVector(sp.targetPose[1], sp.targetPose[2], p.target_height)
    distance_reward = p.r_distance*(norm(sp.uavPose - target3Dpose)+0.00001)
    return action_reward + detection_reward + outScene_reward + distance_reward
end

function POMDPs.generate_sr(pp::UAVchasePOMDP, s::MDPState, a::MDPAction, rng::AbstractRNG)
    sp = generate_s(pp, s, a, rng)
    return sp, reward(pp,a, sp)
end

function POMDPs.generate_o(pp::UAVchasePOMDP, s::MDPState, a::MDPAction, sp::MDPState, rng::AbstractRNG)
    p = mdp(pp)
    if if_in_FOV(pp, sp) # if target is IN camera field of view (FOV) <=> having a measurement
        meas_cov = Matrix(pp.meas_std*Diagonal{Float64}(I, 2))
        measurement_noise = rand(rng, MvNormal(meas_cov))
        targetPose_true_pred = s.targetPose_true + p.dt*p.target_velocity
        meas_location = targetPose_true_pred + measurement_noise # measurement is the target location + noise
        return targetObservationDistribution(meas_location, true)
    else # not in FOV <=> no measurement available
        targetPose_pred = s.targetPose + p.dt*p.target_velocity # a naive predictor when no measurement is available when taget is NOT within FOV
        return targetObservationDistribution(targetPose_pred, false)
    end
end

function POMDPs.observation(pp::UAVchasePOMDP, s::MDPState, a::MDPAction, sp::MDPState, rng::AbstractRNG)
    return generate_o(pp, s, a, sp, rng) # not sure if this function is needed for anything
end

function POMDPModelTools.obs_weight(pp::UAVchasePOMDP, s::MDPState, a::MDPAction, sp::MDPState, o::targetObservationDistribution)
     p = mdp(pp)
     if if_in_FOV(pp, sp)
         if o.if_observed
             y = o.current_observation # true measurement
             y_hat = s.targetPose + p.dt*p.target_velocity # not sure if prediction should be sp
             sigma_square = pp.meas_std
             return 1.0/(2*pi*sigma_square)*exp(-0.5*( (y[1]-y_hat[1])^2 + (y[2]-y_hat[2])^2 )/sigma_square) # assume normal distribution
         else

             return 0.0
         end
     else # if not in FOV
         if o.if_observed
             return 0.0
         else
             return 1.0
         end
     end
 end

function POMDPs.actions(pp::UAVchasePOMDP)
    p = mdp(pp)
    xy = p.UAV_xy_speed
    z = p.UAV_z_speed
    action_list= [MDPAction(xy,0.,0.), MDPAction(xy,0.,15.), MDPAction(xy,0.,30.), MDPAction(xy,0.,45.),
                  MDPAction(0.,0.,0.), MDPAction(xy,0.,-15.), MDPAction(xy,0.,-30.), MDPAction(xy,0.,-45.),
                  MDPAction(0.,z*2/3,0.), MDPAction(0.,-z, 0.), MDPAction(xy,0.,90.), MDPAction(xy,0.,-90.),
                  MDPAction(2*xy,0.,0.), MDPAction(2*xy,0.,15.), MDPAction(2*xy,0.,30.), MDPAction(2*xy,0.,45.),
                  MDPAction(0.,-3*z,0.), MDPAction(2*xy,0.,-15.), MDPAction(2*xy,0.,-30.), MDPAction(2*xy,0.,-45.),
                  MDPAction(0.,-2*z, 0.), MDPAction(2*xy,0.,90.), MDPAction(2*xy,0.,-90.),
                  MDPAction(-xy/2,0.,0.),MDPAction(0.,10*z,0.), MDPAction(0.,20*z,0.), MDPAction(0.,50*z,0.),
                  MDPAction(xy/2,0.,0.), MDPAction(xy/2,0.,-15.), MDPAction(xy/2,0.,30.), MDPAction(xy/2,0.,-45.),
                  MDPAction(xy/3,0.,0.), MDPAction(xy/3,0.,15.), MDPAction(xy/3,0.,-30.), MDPAction(xy/3,0.,45.),
                  MDPAction(xy/6,0.,0.), MDPAction(xy/6,0.,-30.), MDPAction(xy/6,0.,30.),MDPAction(-xy/6,0.,0.)]
    return action_list
end
