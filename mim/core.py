import numpy
from .wrapper import ffi, lib


class Model:
    """Parametric model of images, parametrised by a scalar.
    
       Note: The model is assumed to be continuous w.r.t. the scalar parameter.
    """
    
    def __init__(self, parameter, images):
        # Sanity check
        assert(parameter.dtype == "f8")
        assert(parameter.ndim == 1)
        assert(images.dtype == "f8")
        assert(images.ndim == 3)
        assert(parameter.shape[0] == images.shape[0])
        
        # Allocate the C object.
        c_model = ffi.new('struct mim_model *[1]')
        c_model[0] = lib.mim_model_create_w(
            parameter.strides[0],
            ffi.from_buffer(parameter),
            (*images.shape, *images.strides),
            ffi.from_buffer(images)
        )
        
        # Register a garbage collector for the C object.
        self._c = ffi.gc(
            c_model,
            lambda x: x[0].destroy(x)
        )
    
    @property
    def shape(self):
        """Shape of model images."""
        return (
            int(self._c[0].width),
            int(self._c[0].height)
        )
    
    @property
    def pmin(self):
        """Model parameter lower bound."""
        return float(self._c[0].pmin)
    
    @property
    def pmax(self):
        """Model parameter upper bound."""
        return float(self._c[0].pmax)
    
    def __call__(self, parameter):
        """Get a snapshot of the model for the given parameter value."""
        
        image = numpy.empty(self.shape)
        
        lib.mim_model_get_w(
            self._c[0],
            parameter,
            (*image.shape, *image.strides),
            ffi.from_buffer(image)
        )        
        return image
    
    def invert(self, observation, filter_=None):
        """Get parameter values for a given observation."""
        
        assert(observation.dtype == "f8")
        assert(observation.shape == self.shape)
        
        if filter_ is not None:
            assert(filter_.dtype == "f8")
            assert(filter_.shape == self.shape)
        
        parameter = numpy.empty(self.shape)
        
        if filter_ is None:
            lib.mim_model_invert_w(
                self._c[0],
                (*observation.shape, *observation.strides),
                ffi.from_buffer(observation),
                (*observation.shape, *observation.strides),
                ffi.NULL,
                (*parameter.shape, *parameter.strides),
                ffi.from_buffer(parameter)
            )
        else:
            filter_ = filter_.copy(order='C')
            lib.mim_model_invert_w(
                self._c[0],
                (*observation.shape, *observation.strides),
                ffi.from_buffer(observation),
                (*filter_.shape, *filter_.strides),
                ffi.from_buffer(filter_),
                (*parameter.shape, *parameter.strides),
                ffi.from_buffer(parameter)
            )
        return parameter
    
    def invert_min(self, observation, min_value, filter_=None, sigma=1.0):
        """Get parameter, bin and integrated values for a 
           given observation and a minimum value.
        """
        
        # Sanity check
        assert(observation.dtype == "f8")
        assert(observation.shape == self.shape)
        
        if filter_ is not None:
            assert(filter_.dtype == "f8")
            assert(filter_.shape == self.shape)
        
        parameter = numpy.empty(self.shape)
        bins = numpy.empty(self.shape)
        values = numpy.empty(self.shape)
        
        if filter_ is None:
            lib.mim_model_min_invert_w(
                self._c[0],
                (*observation.shape, *observation.strides),
                ffi.from_buffer(observation),            
                (*observation.shape, *observation.strides),
                ffi.NULL,            
                (*parameter.shape, *parameter.strides),
                (ffi.from_buffer(parameter), ffi.from_buffer(bins), ffi.from_buffer(values)),
                min_value, sigma
            )
        else:
            filter_ = filter_.copy(order='C')  
            lib.mim_model_min_invert_w(
                self._c[0],
                (*observation.shape, *observation.strides),
                ffi.from_buffer(observation),            
                (*filter_.shape, *filter_.strides),
                ffi.from_buffer(filter_),            
                (*parameter.shape, *parameter.strides),
                (ffi.from_buffer(parameter), ffi.from_buffer(bins), ffi.from_buffer(values)),
                min_value, sigma
            )
        
        return parameter, bins, values


class Prng:
    """Pseudo random number generator."""
    
    def __init__(self, seed=0):
        
        # Allocate the C object.
        c_prng = ffi.new('struct mim_prng *[1]')
        
        if seed < 0:
            seed = 0
        
        c_prng[0] = lib.mim_prng_init(seed)
        
        self._c = ffi.gc(
            c_prng,
            lambda x: x[0].destroy(x)
        )
    
    @property
    def seed(self):
        return int(self._c[0].seed)
    
    @property
    def weight(self):
        return float(self._c[0].weight(self._c[0]))
    
    def uniform(self, low=0, up=1):
        u = self._c[0].uniform1(self._c[0])
        return low + u * (up - low)
    
    def normal(self, mu=0, sigma=1):
        return self._c[0].normal(self._c[0], mu, sigma)
    
    def poisson(self, par):
        return self._c[0].poisson(self._c[0], par)
