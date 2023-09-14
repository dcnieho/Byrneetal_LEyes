import math
import random

import numpy as np
import deeptrack as dt
from scipy import stats, interpolate
from scipy.spatial import transform

from image_sizes import CNN_IMAGE_SIZE, EDS2019_IMAGE_SIZE, HOURGLASS_IMAGE_SIZE

## helpers
def drawFromRange(range, n=1):
    # range - list of length 2, e.g., [1, 2]
    return range[0] + np.random.rand(n) * np.diff(range)

def drawTrunc(func,params,lims,n=1):
    def drawTruncInternal(func,params,lims):
        while True:
            val = func(**params)
            if val>lims[0] and val<lims[1]:
                return val
    out = np.zeros((n))
    for i in range(n):
        out[i] = drawTruncInternal(func,params,lims)
    return out

# inverse unnormalized Gaussian for scaling
def gauss1d_inv(v=0, mx=0, sx=1):
    return mx + np.sqrt(2)*sx*np.sqrt(-np.log(v))

# calculate tail width
def tail_width(v):
    # gauss1d_inv(v/2) / gauss1d_inv(v)
    # i.e., tail width is defined as FWHM of truncated Gaussian
    return np.sqrt(np.log(2)/np.log(v)+1)

# inverse of tail width, so we can sample tail widths uniformly
def tail_width_inv(w):
    return 2**(1/(w**2 - 1))

# define 2D Gaussian (without normalization term so peak is 1)
def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

# define 2D Gaussian (without normalization term so peak is 1)
def gauss2d_oriented(x=0, y=0, mx=0, my=0, sx=1, sy=1, theta=0):
    a = np.cos(theta)**2 / (2 * sx**2) + np.sin(theta)**2 / (2 * sy**2);
    b = np.sin(2 * theta) / (4 * sx**2) - np.sin(2 * theta) / (4 * sy**2);
    c = np.sin(theta)**2 / (2 * sx**2) + np.cos(theta)**2 / (2 * sy**2);
    return np.exp(-(a*(x - mx)**2. + 2.*b*(x - mx)*(y - my) + c*(y - my)**2.))

def sigma_generator(nCR,range,ratio_range):
    sigmas = drawFromRange(range, nCR).reshape((nCR,1))
    ratios = drawFromRange(ratio_range, nCR).reshape((nCR,1))
    return np.hstack((sigmas, np.multiply(sigmas,ratios)))

def positionCRs_noOverlap(sigmas, imsize, prob_surface_params=None):
    if not prob_surface_params:
        prob_surface = lambda x, y: 1
    else:
        # NB: this uses rejection sampling.
        # we want positions near pupil center to be unlikely, specifically, we
        # want the 2D density of the positions to be 1-gauss2d_oriented(<pupil parameters>)
        # so for each point we generate an x and y coordinate, and a probability (0-1)
        # if 1-gauss2d_oriented(x,y)<the generated probability, reject the point
        p_position, p_sigmas, p_ori = prob_surface_params
        prob_surface = lambda x, y: 1 - gauss2d_oriented(x, y, p_position[0], p_position[1], 1.5 * p_sigmas[0],
                                                         1.5 * p_sigmas[1], p_ori)
    sigmas = np.amax(sigmas,1)
    nCR = len(sigmas)
    pos = np.empty((nCR,2,))
    pos[:] = np.nan
    for p in range(nCR):
        ok = False
        while not ok:
            ppos = drawFromRange([sigmas[p],imsize-sigmas[p]],2)
            prob = np.random.rand(1)

            if prob_surface(ppos[0], ppos[1]) < prob:
                # reject point due to probability surface
                ok = False
            elif p==0:
                ok = True
            else:
                dists = np.hypot(pos[0:p,0]-ppos[0], pos[0:p,1]-ppos[1])
                ok = all(dists>1.25*(sigmas[0:p]+sigmas[p]))    # 1.25 for little extra space (CRs have tail and even if not we wouldn't want them to touch)
        pos[p,:] = ppos
    return pos

def generate_polygon(avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_verts: int,
                     center = (0,0),
                     rotate_angle = None,
                     num_final_verts = None
                     ):
    # based on https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    spikiness *= avg_radius
    angles = drawFromRange([1-irregularity,1+irregularity],num_verts)
    angles = np.cumsum(angles/np.sum(angles)*2*np.pi)
    if rotate_angle:
        angles += rotate_angle

    # now generate the points
    r = drawTrunc(stats.norm.rvs, {'loc':avg_radius,'scale':spikiness}, [0, 2 * avg_radius], num_verts)
    points = np.reshape(center,(-1,1)) + np.vstack((np.multiply(r,np.cos(angles)), np.multiply(r,np.sin(angles))))
    if num_final_verts is not None and num_final_verts > num_verts:
        # periodic interpolation to subsample shape
        cs_x = interpolate.CubicSpline(np.append([0],angles), np.append([points[0,-1]],points[0,:]))
        cs_y = interpolate.CubicSpline(np.append([0],angles), np.append([points[1,-1]],points[1,:]))
        n_angles = np.linspace(0,2*np.pi-np.pi/num_final_verts/2,num_final_verts)

        return np.vstack((cs_x(n_angles),cs_y(n_angles)))
    else:
        return points

def compose(image, feature, mode, val=0):
    if mode=='add':
        return image+feature
    elif mode=='max':
        return np.maximum(image,feature)
    elif mode=='subtract':
        return np.clip(image-feature,val,np.inf) # make sure we don't go below val
    elif mode=='blend':
        mask = np.clip(feature,0,1)
        return mask*val+(1-mask)*image

## signed distance fields
# ellipses and other shapes using signed distance fields so we can give them
# smooth edges. This is the code for generating those shapes
def center_rot_pos(xy_grid, pos, ori):
    # center and rotate problem
    p = xy_grid-np.reshape(pos,(-1,1))
    if ori is None:
        return p

    cosr = np.cos(ori);
    sinr = np.sin(ori);
    return np.vstack(( cosr * p[0,:] + sinr * p[1,:],
                      -sinr * p[0,:] + cosr * p[1,:]))
def matlab_dot(A,B):
    return np.sum(A.conj()*B, axis=0)

def ellipse(xy_grid, pos, radii, ori=0):
    p = center_rot_pos(xy_grid, pos, ori)
    radii = np.reshape(radii,(-1,1))

    # https://www.shadertoy.com/view/4lsXDN
    # symmetry
    p = np.abs(p)

    # find root with Newton solver
    q = np.multiply(radii,p-radii)
    w = np.zeros((1,q.shape[1]))
    w[:,q[0,:]<q[1,:]] = np.pi/2
    for i in range(6):
        cs = np.vstack((np.cos(w), np.sin(w)))
        u = np.multiply(radii,cs)
        v = np.multiply(radii,np.vstack((-cs[1,:], cs[0,:])))
        w = w + np.divide(matlab_dot(p-u,v), matlab_dot(p-u,u)+matlab_dot(v,v))

    # compute final point and distance
    dist = np.linalg.norm(p-np.multiply(radii, np.vstack((np.cos(w), np.sin(w)))), axis=0)

    # return signed distance
    t = np.divide(p,radii)
    c = matlab_dot(t,t)
    should_flip = c<=1
    dist[should_flip] = -dist[should_flip]
    return dist

def polygon(xy_grid, pos, poly_coords):
    p = center_rot_pos(xy_grid, pos, None)
    num_vert = poly_coords.shape[1]
    d1 = p-poly_coords[:,[1]]
    d = matlab_dot(d1,d1)
    s = np.ones((1,p.shape[1]))
    j = num_vert-1
    for i in range(num_vert):
        # distance
        e = poly_coords[:,[j]] - poly_coords[:,[i]]
        w =                  p - poly_coords[:,[i]]
        b = w - np.multiply(e,np.clip( np.divide(matlab_dot(w,e),matlab_dot(e,e)), 0., 1.))
        d = np.minimum( d, matlab_dot(b,b) )

        # winding number from http://geomalgorithms.com/a03-_inclusion.html
        cond = np.array(
               [p[1,:]>=poly_coords[1,i],
                p[1,:]< poly_coords[1,j],
                np.multiply(e[0,:],w[1,:]) > np.multiply(e[1,:],w[0,:])])
        should_flip = np.logical_or(np.all(cond,axis=0), np.logical_not(np.any(cond,axis=0)))
        s[0,should_flip] = -s[0,should_flip]
        j=i
    return np.multiply(s,np.sqrt(d))

def shape_maker(pos, im_size, shape, smooth_edge, **kwargs):
    im_size = np.array(im_size)
    pos = np.array(pos).flatten()

    pos -= im_size/2
    x,y=np.meshgrid(np.linspace(-im_size[0]/2+.5,im_size[0]/2-.5,int(im_size[0])), np.linspace(-im_size[1]/2+.5,im_size[1]/2-.5,int(im_size[1])))
    xy_grid = np.array([x.T.flatten(), y.T.flatten()])
    if shape=='ellipse':
        dist = ellipse(xy_grid, pos, **kwargs)
    elif shape=='polygon':
        dist = polygon(xy_grid, pos, **kwargs)
    dist = np.reshape(dist,x.shape[::-1]).T

    # add raised cosine edge
    return raised_cosine(dist, smooth_edge)

def raised_cosine(dist, smooth_edge):
    # edge is added outside the object, so object is at full height for entire size you specify
    dist = np.where(dist < 0, 0, dist)          # all internal distances set to 0
    dist = np.minimum(dist / smooth_edge, 1)    # scale ramp such that it goes from 0--1, values outside ramp are set to 1

    # apply raised cosine to ramp
    return np.cos(dist*np.pi)/2+.5

## Chugh et al. house-shaped CR polygon
def generate_house(length, rect_height, roof_height, center, distortion, rotate_angle, base_squeeze, image_size, invalid=-1):
    rect_height = length * rect_height
    roof_height = length * roof_height
    vertices = np.array(create_rectangle(length, rect_height, center, base_squeeze)).squeeze()
    vertices = np.array(distort_polygon(vertices, distortion)).squeeze()
    vertices = np.vstack([perpendicular_point(*vertices[-1], *vertices[0], roof_height[0]), vertices])
    vertices = rotate_points(rotate_angle, center, vertices)

    vertices[(vertices <= 0).any(1)] = np.array([invalid, invalid])
    vertices[(vertices[:, 0] >= image_size)] = np.array([invalid, invalid])
    vertices[(vertices[:, 1] >= image_size)] = np.array([invalid, invalid])
    return vertices

def perpendicular_point(x1, y1, x2, y2, height):
    # coordinates of the midpoint of the base
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # coordinates of the top point of the triangle
    top_x = mid_x
    top_y = mid_y - height
    return top_x, top_y

def create_rectangle(length, height, center, base_squeeze):
    squeeze = length * base_squeeze
    # Calculate the coordinates of the top-left corner of the rectangle
    x = center[0] - length / 2
    y = center[1] - height / 2

    # Create a list of four points representing the corners of the rectangle
    points = [(x + length, y), (x + length - squeeze, y + height), (x+squeeze, y + height), (x, y)]
    return points

def distort_polygon(points, magnitude):
    # Generate a random vector for each point in the polygon
    vectors = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(len(points))]

    # Calculate the magnitude of each vector
    magnitudes = [magnitude * math.sqrt(v[0] ** 2 + v[1] ** 2) for v in vectors]

    # Normalize the vectors to the desired magnitude
    vectors = [(v[0]/m, v[1]/m) if m > 0 else v for v, m in zip(vectors, magnitudes)]

    # Apply the distortion to each point in the polygon
    distorted_points = [(p[0] + v[0], p[1] + v[1]) for p, v in zip(points, vectors)]
    return distorted_points

def rotate_points(angle, center, vertices):
    r = transform.Rotation.from_euler('z', angle, degrees=True)
    vertices = np.subtract(np.array(vertices), center)
    vertices = r.apply(np.pad(vertices, (0, 1))[:-1])[:, :-1]
    vertices = np.add(vertices, center)
    return vertices

## features
class GrayBackground(dt.Feature):
    __list_merge_strategy__ = dt.MERGE_STRATEGY_APPEND
    __distributed__ = False

    def __init__(self, image_size, **kwargs):
        self.image_size = image_size
        super().__init__(**kwargs)

    def get(self, image, luminance, **kwargs):
        bg = np.ones((self.image_size,self.image_size),dtype=np.float64)*luminance*255
        return bg

class GradientBackground(dt.Feature):
    # These two are required to make sure that this background can start a pipeline
    __list_merge_strategy__ = dt.MERGE_STRATEGY_APPEND
    __distributed__ = False

    def __init__(self, image_size, **kwargs):
        self.image_size = image_size
        super().__init__(**kwargs)

    def get(self, image, lum1, lum2, ori, **kwargs):
        x = np.arange(self.image_size)
        y = np.arange(self.image_size)
        X, Y = np.meshgrid(y, x)

        mlum = min([lum1, lum2])
        dlum = (max([lum1, lum2]) - mlum) / self.image_size
        gradient = [np.cos(ori) * dlum, -np.sin(ori) * dlum]
        lum = X * gradient[0] + Y * gradient[1] + mlum

        return lum*255

class TwoToneBackground(dt.Feature):
    __list_merge_strategy__ = dt.MERGE_STRATEGY_APPEND
    __distributed__ = False

    def __init__(self, image_size, **kwargs):
        self.image_size = image_size
        super().__init__(**kwargs)

    def get(self, image, pos, ori, lum, bg_lum, smooth_edge, **kwargs):
        x = np.arange(0, self.image_size)+((self.image_size+1)%2)/2
        y = np.arange(0, self.image_size)+((self.image_size+1)%2)/2
        x, y = np.meshgrid(x, y)

        x -= pos[0]
        y -= pos[1]

        rx = np.cos(ori) * x + np.sin(ori) * y
        rx = np.maximum(np.minimum(rx,smooth_edge/2),-smooth_edge/2)/smooth_edge+.5

        # apply raised cosine, from background to foreground luminance
        return (np.cos(rx*np.pi)/2+.5)*(lum-bg_lum)+bg_lum

class Gaussian(dt.Feature):
    def get(self, image, position, sigma, dropout=False, **kwargs):
        if dropout:
            return image

        x = np.arange(0, image.shape[1])
        y = np.arange(0, image.shape[0])
        x, y = np.meshgrid(x, y)

        # generate
        feature = gauss2d(x, y, mx=position[0], my=position[1], sx=sigma[0], sy=sigma[1]) * 255
        return np.maximum(image, feature)

class SaturatedGaussian(dt.Feature):
    def get(self, image, position, sigma, gauss_amp, ori = 0., val=1., type='CR', mode=None, dropout = False, **kwargs):
        if dropout:
            # don't draw feature
            return image

        # get positions and sigmas to use
        if 'CRs' in type:
            sigma=np.array([sigma[kwargs['_ID'][0],:]])
            position=position[kwargs['_ID'][0]]
        if not isinstance(sigma,np.ndarray) or sigma.size==1:
            sigma = np.array([sigma, sigma])
        sigma = sigma.flatten()

        # good default for mode based on type of feauture
        if mode is None:
            if 'CR' in type:
                mode = 'max'
            elif 'pupil' in type:
                mode = 'subtract'

        # determine which Gaussian to use
        if 'oriented' in type:
            g_func = gauss2d_oriented
            args = {'theta':ori}
        else:
            g_func = gauss2d
            args = {}

        # prep gen
        x = np.arange(0, image.shape[1])
        y = np.arange(0, image.shape[0])
        x, y = np.meshgrid(x, y)
        fac = gauss1d_inv(1/gauss_amp)
        sd = sigma/fac

        # generate
        feature = gauss_amp*g_func(x,y,mx=position[0], my=position[1], sx=sd[0], sy=sd[1], **args)
        if mode!="blend":
            feature*=255

        # add to image
        return compose(image, feature, mode, val)

class Ellipse(dt.Feature):
    def get(self, image, position, radii, edge_width, val, mode='blend', ori=0, **kwargs):
        feature = shape_maker(position, (image.shape[0],image.shape[1]), 'ellipse', edge_width, radii=radii, ori=ori)
        return compose(image, feature, mode, val)

class Polygon(dt.Feature):
    def get(self, image, position, poly_coords, edge_width, val, mode='blend', **kwargs):
        feature = shape_maker(position, (image.shape[0],image.shape[1]), 'polygon', edge_width, poly_coords=poly_coords)
        return compose(image, feature, mode, val)

class Discretize(dt.Feature):
    def get(self, image, nbit, **kwargs):
        if np.isinf(nbit):
            return image
        else:
            fac = 2**nbit-1
            return np.round(image*fac)/fac

## pipelines
def get_CR_pipeline(stage, freq):
    # parameters
    IMAGE_SIZE = CNN_IMAGE_SIZE

    CR_SIGMA_RANGE = [1, 30]
    GAUSS_AMPLITUDE_RANGE = [2, 20000]

    NOISE_SD_RANGE = [0, 30]

    POSITION_RANGE = 1.5

    BACKGROUND_LUM_EXP_LOC = 1.
    BACKGROUND_LUM_EXP_SCALE = 10.
    # for 1000 hz:
    BG_LUMINANCE_RANGE = [.125, .6]

    # feature instances
    sim_args = dt.Arguments(
        cr_sigma   =lambda: drawFromRange(CR_SIGMA_RANGE),
        cr_position=lambda cr_sigma: drawFromRange([cr_sigma[0],IMAGE_SIZE-cr_sigma[0]],2) if stage==1 else (np.random.rand(2)-.5) * POSITION_RANGE + IMAGE_SIZE/2,
        background_pos=lambda cr_position, cr_sigma: [np.random.normal(cr_position[0], 1.5*cr_sigma), np.random.normal(cr_position[1], 1.5*cr_sigma)]
        )
    background = TwoToneBackground(
        image_size = IMAGE_SIZE,
        **sim_args.properties,
        pos=lambda background_pos: background_pos,
        ori=lambda: drawFromRange([0,2*np.pi]),
        lum=lambda: 255/2 if freq==500 else drawFromRange(BG_LUMINANCE_RANGE)*255,
        bg_lum=lambda: stats.expon.rvs(loc=BACKGROUND_LUM_EXP_LOC,scale=BACKGROUND_LUM_EXP_SCALE),
        smooth_edge = 1 if freq==500 else 4
    ).bind_arguments(sim_args)
    CR = SaturatedGaussian(
        **sim_args.properties,
        sigma=lambda cr_sigma: cr_sigma,
        position=lambda cr_position: cr_position,
        gauss_amp=lambda: drawFromRange(GAUSS_AMPLITUDE_RANGE),
        type='CR'
    ).bind_arguments(sim_args)
    discretizer = Discretize(nbit=8)

    # image generation pipeline
    image_pipeline = background >> CR
    image_pipeline >>= dt.Gaussian(sigma=lambda: drawFromRange(NOISE_SD_RANGE))
    image_pipeline >>= dt.math.Clip(min=0., max=255.) >> dt.NormalizeMinMax(0,1) >> discretizer

    # make data pipeline: image + labels
    def get_position(image):
        return np.array(image.get_property("position"))
    data_pipeline = image_pipeline & (image_pipeline >> get_position)

    return IMAGE_SIZE, data_pipeline, image_pipeline

def get_pupil_pipeline(stage, freq):
    # parameters
    IMAGE_SIZE = CNN_IMAGE_SIZE

    BG_LUMINANCE_RANGE = [.25, .7] if freq==500 else [.125, .6]

    GAUSS_AMPLITUDE_RANGE = [2, 20000]

    PUPIL_SIGMA_RANGE = [20, 60]
    PUPIL_SIGMA_RATIO = [1, 1.3]
    PUPIL_LUM_EXP_LOC = 1.
    PUPIL_LUM_EXP_SCALE = 10.
    POSITION_RANGE = 1.5

    MAX_N_CRs = 4 if stage==1 else 1
    CR_SIGMA_RANGE = [4, 12]
    CR_SIGMA_RATIO = [1, 1.1]

    NOISE_SD_RANGE = [0, 30]

    # for 1000 Hz:
    PUPIL_LUM_LIMS = [0, 25.5]

    # feature instances
    sim_args = dt.Arguments(
        N_CRs=lambda: np.random.randint(1,MAX_N_CRs+1),
        all_CR_sigmas=lambda N_CRs: sigma_generator(N_CRs, CR_SIGMA_RANGE, CR_SIGMA_RATIO),
        all_CR_positions=lambda all_CR_sigmas: positionCRs_noOverlap(all_CR_sigmas, IMAGE_SIZE),
        )
    background = GrayBackground(
        image_size = IMAGE_SIZE,
        luminance=lambda: drawFromRange(BG_LUMINANCE_RANGE),
    )
    pupil = SaturatedGaussian(
        pupil_sigma=lambda: sigma_generator(1, PUPIL_SIGMA_RANGE, PUPIL_SIGMA_RATIO),
        sigma=lambda pupil_sigma: pupil_sigma,
        max_sigma=lambda sigma: np.max(sigma),
        pupil_position=lambda max_sigma: drawFromRange([max_sigma*1.1,IMAGE_SIZE-max_sigma*1.1],2) if stage==1 else (np.random.rand(2)-.5) * POSITION_RANGE + IMAGE_SIZE/2,
        position=lambda pupil_position: pupil_position,
        gauss_amp=lambda: drawFromRange(GAUSS_AMPLITUDE_RANGE),
        ori=lambda: drawFromRange([0, 2*np.pi]),
        pupil_lum=lambda: stats.expon.rvs(loc=PUPIL_LUM_EXP_LOC,scale=PUPIL_LUM_EXP_SCALE) if freq==500 else drawTrunc(stats.expon.rvs, {'loc':PUPIL_LUM_EXP_LOC,'scale':PUPIL_LUM_EXP_SCALE},PUPIL_LUM_LIMS),
        val=lambda pupil_lum: pupil_lum,
        type='pupil_oriented'
    )
    CR = SaturatedGaussian(
        **sim_args.properties,
        sigma=lambda all_CR_sigmas: all_CR_sigmas,
        position=lambda all_CR_positions: all_CR_positions,
        gauss_amp=lambda: drawFromRange(GAUSS_AMPLITUDE_RANGE),
        ori=lambda: drawFromRange([0, 2*np.pi]),
        type='CRs_oriented'
    ).bind_arguments(sim_args)
    CRs = dt.Repeat(
        **sim_args.properties,
        feature=CR,
        N=lambda N_CRs: N_CRs
    ).bind_arguments(sim_args)
    discretizer = Discretize(nbit=8)

    # image generation pipeline
    image_pipeline = background >> pupil >> CRs
    image_pipeline >>= dt.Gaussian(sigma=lambda: drawFromRange(NOISE_SD_RANGE))
    image_pipeline >>= dt.math.Clip(min=0., max=255.) >> dt.NormalizeMinMax(0,1) >> discretizer

    # make data pipeline: image + labels
    def get_position(image):
        return np.array(image.get_property("pupil_position"))
    data_pipeline = image_pipeline & (image_pipeline >> get_position)

    return IMAGE_SIZE, data_pipeline, image_pipeline

def get_EDS2019_pipeline():
    IMAGE_SIZE = EDS2019_IMAGE_SIZE

    # sclera is background
    SCLERA_LUM_MEAN    = .85
    SCLERA_LUM_SD      = .1
    SCLERA_LUM_LIMS    = [.65, 1.]

    IRIS_SIGMA_RANGE = [30, 42.5]
    IRIS_SIGMA_RATIO = [1, 1.3]
    IRIS_EDGE_RANGE = [8, 20]
    IRIS_LUM_LIMS = [.17*255, .8*255]
    IRIS_LUM_MEAN = 77
    IRIS_LUM_SD = 16

    COLLARETTE_RADIUS_FAC = [.3, .6]
    COLLARETTE_POS_SIGMA_FAC = .15
    COLLARETTE_CORNERS = [13,24]
    COLLARETTE_SPIKINESS = [0.05,0.2]
    COLLARETTE_IRREGULARITY = 0.4
    COLLARETTE_CONTRAST = [.25, .6]
    COLLARETTE_EDGE_RANGE = [1,4]

    PUPIL_SIGMA_RANGE = [10, 30]
    PUPIL_SIGMA_RATIO = [1, 1.3]
    PUPIL_LUM_NORM_MEAN = 34
    PUPIL_LUM_NORM_SD = 15
    PUPIL_LUM_LIMS = [0, 65]
    PUPIL_GAUSS_AMP_RANGE = [2, 2000]

    MAX_N_CRs = 8
    CR_SIGMA_RANGE = [.8, 4]
    CR_SIGMA_RATIO = [1, 1.4]
    CR_GAUSS_AMP_RANGE = [2, 20000]

    NOISE_SD_RANGE = [0, 15]

    sim_args = dt.Arguments(
        pupil_sigma=lambda: sigma_generator(1, PUPIL_SIGMA_RANGE, PUPIL_SIGMA_RATIO),
        max_pupil_sigma=lambda pupil_sigma: np.max(pupil_sigma),
        pupil_position=lambda max_pupil_sigma: drawFromRange([max_pupil_sigma*1.1,IMAGE_SIZE-max_pupil_sigma*1.1],2),
        pupil_lum=lambda: drawTrunc(stats.norm.rvs, {'loc':PUPIL_LUM_NORM_MEAN,'scale':PUPIL_LUM_NORM_SD},PUPIL_LUM_LIMS),

        iris_sigmas=lambda: sigma_generator(1, IRIS_SIGMA_RANGE, IRIS_SIGMA_RATIO),
        iris_max_sigma=lambda iris_sigmas: np.max(iris_sigmas),
        iris_position=lambda iris_max_sigma: drawFromRange([iris_max_sigma*1.1,IMAGE_SIZE-iris_max_sigma*1.1],2),
        iris_lum=lambda pupil_lum: drawTrunc(stats.norm.rvs, {'loc':IRIS_LUM_MEAN,'scale':IRIS_LUM_SD},[pupil_lum+.05*255, IRIS_LUM_LIMS[1]]),

        collarette_position=lambda iris_position, iris_max_sigma: [stats.norm.rvs(iris_position[0],iris_max_sigma*COLLARETTE_POS_SIGMA_FAC), stats.norm.rvs(iris_position[1],iris_max_sigma*COLLARETTE_POS_SIGMA_FAC)],
        collarette_contrast=lambda: drawFromRange(COLLARETTE_CONTRAST),
        collarette_lum=lambda iris_lum, collarette_contrast: iris_lum*(1+collarette_contrast),

        N_CRs=lambda: np.random.randint(1,MAX_N_CRs+1),
        CR_sigmas=lambda N_CRs: sigma_generator(N_CRs, CR_SIGMA_RANGE, CR_SIGMA_RATIO),
        CR_positions=lambda CR_sigmas: positionCRs_noOverlap(CR_sigmas, IMAGE_SIZE),
    )

    sclera = GrayBackground(
        image_size = IMAGE_SIZE,
        luminance = lambda: drawTrunc(stats.norm.rvs, {'loc':SCLERA_LUM_MEAN,'scale':SCLERA_LUM_SD}, SCLERA_LUM_LIMS),
    )

    iris = Ellipse(
        **sim_args.properties,
        edge_width=lambda: drawFromRange(IRIS_EDGE_RANGE),
        position=lambda iris_position: iris_position,
        radii=lambda iris_sigmas: iris_sigmas,
        ori=lambda: drawFromRange([0, 2*np.pi]),
        val=lambda iris_lum: iris_lum,
    ).bind_arguments(sim_args)

    collarette = Polygon(
        **sim_args.properties,
        edge_width=lambda: drawFromRange(COLLARETTE_EDGE_RANGE),
        position=lambda collarette_position: collarette_position,
        num_verts=lambda: np.random.randint(COLLARETTE_CORNERS[0],COLLARETTE_CORNERS[1]+1),
        avg_radius=lambda iris_max_sigma: drawFromRange([COLLARETTE_RADIUS_FAC[0]*iris_max_sigma,COLLARETTE_RADIUS_FAC[1]*iris_max_sigma]),
        spikiness=lambda: drawFromRange(COLLARETTE_SPIKINESS),
        poly_coords=lambda avg_radius,num_verts,spikiness: generate_polygon(avg_radius=avg_radius,
                                irregularity=COLLARETTE_IRREGULARITY,
                                spikiness=spikiness,
                                num_verts=num_verts,
                                num_final_verts=num_verts*5),
        val=lambda collarette_lum: collarette_lum
    ).bind_arguments(sim_args)

    pupil = SaturatedGaussian(
        **sim_args.properties,
        sigma=lambda pupil_sigma: pupil_sigma,
        position=lambda pupil_position: pupil_position,
        pupil_ori=lambda: drawFromRange([0, 2*np.pi]),
        pupil_gauss_amp=lambda: tail_width_inv(drawFromRange([tail_width(a) for a in PUPIL_GAUSS_AMP_RANGE])),
        mode='blend',
        type='pupil_oriented',
        ori = lambda pupil_ori: pupil_ori,
        val = lambda pupil_lum: pupil_lum,
        gauss_amp = lambda pupil_gauss_amp: pupil_gauss_amp
    ).bind_arguments(sim_args)

    CR = SaturatedGaussian(
        **sim_args.properties,
        sigma=lambda CR_sigmas: CR_sigmas,
        position=lambda CR_positions: CR_positions,
        ori=lambda: drawFromRange([0, 2*np.pi]),
        type='CRs_oriented',
        gauss_amp=lambda: tail_width_inv(drawFromRange([tail_width(a) for a in CR_GAUSS_AMP_RANGE]))
    ).bind_arguments(sim_args)

    CRs = dt.Repeat(
        **sim_args.properties,
        feature=CR,
        N=lambda N_CRs: N_CRs
    ).bind_arguments(sim_args)

    discretizer = Discretize(nbit=8)

    # image generation pipeline
    image_pipeline = sclera >> iris >> collarette >> pupil >> CRs
    image_pipeline >>= dt.Gaussian(sigma=lambda: drawFromRange(NOISE_SD_RANGE))
    image_pipeline >>= dt.math.Clip(min=0., max=255.) >> dt.NormalizeMinMax(0,1) >> discretizer

    # make segmentation pipeline
    def get_mask(image):
        def create_mask(dict):
            m = GrayBackground(image_size=IMAGE_SIZE).get(None, luminance=0)
            m = SaturatedGaussian().get(m
                                    , position=dict.get('pupil_position')
                                    , sigma=dict.get('pupil_sigma')
                                    , gauss_amp=dict.get('pupil_gauss_amp')
                                    , ori=dict.get('pupil_ori')
                                    , type='pupil_oriented'
                                    , mode='max'
                                    )
            m = dt.math.Clip().get(m, min=image.get_property('min'), max=image.get_property('max'))
            m = dt.NormalizeMinMax().get(m, 0, 1)
            m = Discretize().get(m, nbit=8)
            m = m >= 1
            return  m

        for k in image.properties:
            if k.get('type')=='pupil_oriented':
                return np.array(create_mask(k), dtype=int)

    segment_pipeline = image_pipeline & (image_pipeline >> get_mask)

    return IMAGE_SIZE, segment_pipeline, image_pipeline

def get_hourglass_pipeline(dataset, stage):
    match dataset:
        case 'Chugh':
            isChugh = True
        case 'EDS2020':
            isChugh = False
        case _:
            raise ValueError(f'get_hourglass_pipeline: dataset=={dataset} not understood')

    IMAGE_SIZE = HOURGLASS_IMAGE_SIZE

    BG_LUMINANCE_RANGE = [.25, .7]

    GAUSS_AMPLITUDE_RANGE = [200, 100000]

    PUPIL_SIGMA_RANGE = [6, 22.5]
    PUPIL_SIGMA_RATIO = [1, 1.3]

    if isChugh:
        PUPIL_EXP_LOC = 1.
        PUPIL_EXP_SCALE = 10.
    else:
        PUPIL_LUM_WEI_C = 2
        PUPIL_LUM_WEI_LOC = 18
        PUPIL_LUM_WEI_SCALE = 25

    # CRs
    N_CRs = 5 if isChugh else 8
    CR_SIGMA_RANGE = [1, 2.5]
    CR_SIGMA_RATIO = [1, 1.1]

    CR_NORMAL_STD_X, CR_NORMAL_STD_Y = 20, 10   # for determining offset of center of CR polygon from pupil center
    if isChugh:
        RECT_WIDTH_RANGE = [IMAGE_SIZE*0.1, IMAGE_SIZE*0.45]
        RECT_HEIGHT_RANGE = [0.5, 0.6]
        ROOF_HEIGHT_RANGE = [0.2, 0.5]
        BASE_SQUEEZE_RANGE = [0.05, 0.2]
        POLYGON_ROTATION_RANGE     = [-45, 45] if stage==1 else [-35, 35]
        POLYGON_IRREGULARITY_RANGE = [0.8, 1] if stage==1 else [.1, 0.8]
        CR_DROPOUT = 0.16 if stage==1 else 0.1
    else:
        POLYGON_IRREGULARITY_RANGE = [0.001, .01] if stage==1 else [0.0001, .01]
        POLYGON_RADIUS_RANGE = [IMAGE_SIZE*0.15, IMAGE_SIZE*0.4]
        POLYGON_ROTATION_RANGE = [-.01, .01]
        CR_DROPOUT = 0.2

    MAX_NUM_DUMMIES = 5 if stage==1 else 3
    DUMMY_SIGMA_RANGE = CR_SIGMA_RANGE
    DUMMY_SIGMA_RATIO = [1, 2.5]

    NOISE_SD_RANGE = [0, 30]

    KEYPOINT_SIGMA = 5 if (not isChugh and stage==1) else 10

    d = 1.1
    offset_x = 0 if (isChugh and stage==1) else 20
    pupil_args = dt.Arguments(
        pupil_sigma=lambda: sigma_generator(1, PUPIL_SIGMA_RANGE, PUPIL_SIGMA_RATIO).flatten(),
        max_pupil_sigma=lambda pupil_sigma: np.max(pupil_sigma),
        pupil_position=lambda max_pupil_sigma: drawFromRange([max_pupil_sigma * d + offset_x, IMAGE_SIZE - max_pupil_sigma * d - offset_x], 2),
        pupil_lum=lambda: stats.expon.rvs(loc=PUPIL_EXP_LOC,scale=PUPIL_EXP_SCALE) if isChugh else stats.weibull_min.rvs(c=PUPIL_LUM_WEI_C, loc=PUPIL_LUM_WEI_LOC, scale=PUPIL_LUM_WEI_SCALE),
        pupil_ori=lambda: drawFromRange([0, 2 * np.pi])
    )
    if isChugh:
        CR_args = dt.Arguments(
            **pupil_args.properties,
            CR_sigmas=lambda: sigma_generator(N_CRs, CR_SIGMA_RANGE, CR_SIGMA_RATIO),
            CR_constellation_position = lambda pupil_position: pupil_position+[np.random.normal(scale=CR_NORMAL_STD_X), np.random.normal(scale=CR_NORMAL_STD_Y)],
            CR_poly_rotation=lambda: drawFromRange(POLYGON_ROTATION_RANGE),
            CR_poly_irregularity=lambda: drawFromRange(POLYGON_IRREGULARITY_RANGE)[0],

            CR_height=lambda: drawFromRange(RECT_HEIGHT_RANGE),
            CR_roof_height=drawFromRange(ROOF_HEIGHT_RANGE),
            CR_width=lambda: drawFromRange(RECT_WIDTH_RANGE),
            CR_base_squeeze=lambda: drawFromRange(BASE_SQUEEZE_RANGE),
            CR_positions= lambda CR_constellation_position, CR_poly_irregularity, CR_poly_rotation, CR_height, CR_roof_height, CR_width, CR_base_squeeze:
                generate_house(center= CR_constellation_position
                               , rect_height=CR_height
                               , roof_height=CR_roof_height
                               , length=CR_width
                               , base_squeeze=CR_base_squeeze
                               , distortion=CR_poly_irregularity
                               , rotate_angle=CR_poly_rotation
                               , image_size=IMAGE_SIZE)
        ).bind_arguments(pupil_args)
    else:
        CR_args = dt.Arguments(
            **pupil_args.properties,
            CR_sigmas=lambda: sigma_generator(N_CRs, CR_SIGMA_RANGE, CR_SIGMA_RATIO),
            CR_constellation_position = lambda pupil_position: pupil_position+[np.random.normal(scale=CR_NORMAL_STD_X), np.random.normal(scale=CR_NORMAL_STD_Y)],
            CR_poly_rotation=lambda: drawFromRange(POLYGON_ROTATION_RANGE),
            CR_poly_irregularity=lambda: drawFromRange(POLYGON_IRREGULARITY_RANGE)[0],

            CR_poly_radius=lambda: drawFromRange(POLYGON_RADIUS_RANGE),
            CR_positions=lambda CR_constellation_position, CR_poly_radius, CR_poly_irregularity, CR_poly_rotation:
                generate_polygon(avg_radius=CR_poly_radius
                                , rotate_angle=CR_poly_rotation
                                , irregularity=CR_poly_irregularity
                                , spikiness=0.01
                                , num_verts=N_CRs
                                , center=CR_constellation_position).T
        ).bind_arguments(pupil_args)

    dummy_args = dt.Arguments(
        **pupil_args.properties,
        N_dummies=lambda: np.random.randint(1,MAX_NUM_DUMMIES+1),
        dummy_sigmas=lambda N_dummies: sigma_generator(N_dummies, DUMMY_SIGMA_RANGE, DUMMY_SIGMA_RATIO),
        dummy_positions=lambda dummy_sigmas, pupil_position, pupil_sigma, pupil_ori: positionCRs_noOverlap(dummy_sigmas, IMAGE_SIZE, [pupil_position, pupil_sigma, pupil_ori])
    ).bind_arguments(pupil_args)

    background = GradientBackground(
        image_size = IMAGE_SIZE,
        lum1=lambda: drawFromRange(BG_LUMINANCE_RANGE),
        lum2=lambda: drawFromRange(BG_LUMINANCE_RANGE),
        ori=lambda: drawFromRange([0, 2*np.pi])
    )

    pupil = SaturatedGaussian(
        **pupil_args.properties,
        sigma=lambda pupil_sigma: pupil_sigma,
        position=lambda pupil_position: pupil_position,
        pupil_gauss_amp=lambda: tail_width_inv(drawFromRange([tail_width(a) for a in GAUSS_AMPLITUDE_RANGE])),
        type='pupil_oriented',
        ori = lambda pupil_ori: pupil_ori,
        val = lambda pupil_lum: pupil_lum,
        gauss_amp = lambda pupil_gauss_amp: pupil_gauss_amp
    ).bind_arguments(pupil_args)

    CR = SaturatedGaussian(
        **CR_args.properties,
        sigma=lambda CR_sigmas: CR_sigmas,
        position=lambda CR_positions: CR_positions,
        ori=lambda: drawFromRange([0, 2*np.pi]),
        type='CRs_oriented',
        lbl='CR',
        gauss_amp=lambda: tail_width_inv(drawFromRange([tail_width(a) for a in GAUSS_AMPLITUDE_RANGE])),
        dropout=lambda: np.random.uniform() < CR_DROPOUT
    ).bind_arguments(CR_args)

    CRs = dt.Repeat(
        **CR_args.properties,
        feature=CR,
        N=N_CRs
    ).bind_arguments(CR_args)

    dummy = SaturatedGaussian(
        **dummy_args.properties,
        sigma=lambda dummy_sigmas: dummy_sigmas,
        position=lambda dummy_positions: dummy_positions,
        ori=lambda: drawFromRange([0, 2*np.pi]),
        type='CRs_oriented',
        lbl='dummy',
        gauss_amp=lambda: tail_width_inv(drawFromRange([tail_width(a) for a in GAUSS_AMPLITUDE_RANGE]))
    ).bind_arguments(dummy_args)

    dummies = dt.Repeat(
        **dummy_args.properties,
        feature=dummy,
        N=lambda N_dummies: N_dummies
    ).bind_arguments(dummy_args)

    discretizer = Discretize(nbit=8)

    # image generation pipeline
    image_pipeline = background >> pupil >> CRs >> dummies
    image_pipeline >>= dt.Gaussian(sigma=lambda: drawFromRange(NOISE_SD_RANGE))
    image_pipeline >>= dt.math.Clip(min=0., max=255.) >> dt.NormalizeMinMax(0,1) >> discretizer

    # make data pipeline: image + feature locations
    def get_position(image):
        # Get positions of features
        positions = []
        for k in image.properties:
            pos = None
            if k.get('type')=='pupil_oriented':
                pos = k.get('position')
            elif k.get('type')=='CRs_oriented' and k.get('lbl')=='CR':
                pos = k.get('position')[k.get('_ID')[0]]
            if pos is not None:
                positions.append(np.reshape(pos,(2,1)))
        return np.array(positions).flatten()

    data_pipeline = image_pipeline & (image_pipeline >> get_position)

    # make segmentation pipeline: separate each object in its own mask
    def get_label(image):
        def create_mask(position, drop = False):
            m = GrayBackground(image_size=IMAGE_SIZE).get(None, luminance=0)
            m = Gaussian().get(m
                                , position=position
                                , sigma=[KEYPOINT_SIGMA, KEYPOINT_SIGMA]
                                , dropout=drop if isChugh else False
            )
            m = dt.math.Clip().get(m, min=image.get_property('min'), max=image.get_property('max'))
            m = dt.NormalizeMinMax().get(m, 0, 1)
            m = Discretize().get(m, nbit=8)

            return m

        multimask = []
        for k in image.properties:
            if k.get('type')=='pupil_oriented':
                multimask.append(create_mask(k.get('position')))
            elif k.get('type')=='CRs_oriented' and k.get('lbl')=='CR':
                multimask.append(create_mask(k.get('position')[k.get('_ID')[0]], k.get('dropout')))
        return np.array(multimask)

    segment_pipeline = image_pipeline & (image_pipeline >> get_label)

    return IMAGE_SIZE, N_CRs, segment_pipeline, data_pipeline, image_pipeline