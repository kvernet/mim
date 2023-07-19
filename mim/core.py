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
    
    def invert(self, observation):
        """Get parameter values for a given observation."""
        
        assert(observation.dtype == "f8")
        assert(observation.shape == self.shape)
        
        parameter = numpy.empty(self.shape)
        
        lib.mim_model_invert_w(
            self._c[0],
            (*observation.shape, *observation.strides),
            ffi.from_buffer(observation),
            (*parameter.shape, *parameter.strides),
            ffi.from_buffer(parameter)
        )
        return parameter
