import numpy as np
import xarray as xr
from functools import wraps


def _rle_filter_extreme_durations(mask, min_duration=5, max_gap=2, n_jobs=36):
    """
    Allow only extreme events that meet a minimum duration requirement, allowing
    for gaps. 
    
    Uses run length encoding to ensure that events meet the required lengths. 
    The default configuration is set to match the set-up from Hobday et al (2018).
    
    Parameters
    ----------
    da : xr.DataArray
        the data array where `time` must be one of the dimensions. The function
        will transpose if time is not the first dimension. 
    min_duration : int [5]
        the minimum duration of an event (days/months). Events shorter than this 
        duration will be dropped. Any even numbers will be rounded up so that
        there is a middle value. 
    max_gap : int [3]
        the maximum duration of a gap between events so that the event is 
        considered one event (in units of the time step).
    n_jobs : int [36]
        the number of jobs that the process will be run on (note that this only
        works when run with _run_blocks)
        
    Returns
    -------
    xr.DataArray 
        has exactly the same dims and coords as the input array, with some 
        changes to the filtering steps. 
        
    Note
    ----
    This run length encoding is considerably slower than using the ndimage
    morphology approach (x50). However this may not be true when min_duration or
    max_gap are large (>25). This approach is a useful test to see if the 
    ndimage morphology approach is achieving the correct result.
    """
    def _rlencode(x, dropna=False):
        """
        Run length encoding.
        Based on http://stackoverflow.com/a/32681075, which is based on the rle 
        function from R.

        Parameters
        ----------
        x : 1D array_like
            Input array to encode
        dropna: bool, optional
            Drop all runs of NaNs.

        Returns
        -------
        start positions, run lengths, run values

        """
        where = np.flatnonzero
        x = np.asarray(x)
        n = len(x)
        if n == 0:
            return (np.array([], dtype=int), 
                    np.array([], dtype=int), 
                    np.array([], dtype=x.dtype))

        starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
        lengths = np.diff(np.r_[starts, n])
        values = x[starts]

        if dropna:
            mask = ~np.isnan(values)
            starts, lengths, values = starts[mask], lengths[mask], values[mask]

        return starts, lengths, values

    def _rldecode(starts, lengths, values, minlength=None):
        """
        Decode a run-length encoding of a 1D array.

        Parameters
        ----------
        starts, lengths, values : 1D array_like
            The run-length encoding.
        minlength : int, optional
            Minimum length of the output array.

        Returns
        -------
        1D array. Missing data will be filled with NaNs.

        """
        starts, lengths, values = map(np.asarray, (starts, lengths, values))
        # TODO: check validity of rle
        ends = starts + lengths
        n = ends[-1]
        if minlength is not None:
            n = max(minlength, n)
        x = np.full(n, np.nan)
        for lo, hi, val in zip(starts, ends, values):
            x[lo:hi] = val
        return x

    def _rle_filter_extreme_durations_arr(arr, min_duration=5, max_gap=2):
        """
        Parameters
        ----------
        arr : np.ndarray
            1D boolean array representing extreme events
        min_duration : int [5]
            minimum duration of an extreme event 
        max_gap : int [2]
            maximum gap between two events for those two events to be considered
            one extreme event
        
        Returns
        -------
        np.ndarray
            a filtered extremes mask
        """
        start, counts, values = _rlencode(arr)
        filling_gaps = (counts <= max_gap) & (values == False)
        values[filling_gaps] = True
        arr2 = _rldecode(start, counts, values).astype(bool)

        start, counts, values = _rlencode(arr2)
        short_extremes = (counts < min_duration) & (values == True)
        values[short_extremes] = False
        arr3 = _rldecode(start, counts, values).astype(bool)

        return arr3
    mask_stacked = mask.stack(coord=['lat', 'lon']).transpose('coord', 'time')
    mask_vals = mask_stacked.values

    # Make sure that extreme events don't leak accross time slices - make first 10 days not extreme
    mask_vals[:, :min_duration+2] = False

    # First dimension sets the number of jobs that will be run
    mask_flat = mask_vals.reshape(-1)
    
    filtered_stacked = _rle_filter_extreme_durations_arr(
        mask_flat, min_duration, max_gap).reshape(mask_vals.shape)
    
    filtered = xr.DataArray(
        filtered_stacked, 
        coords=mask_stacked.coords,
        dims=mask_stacked.dims).unstack()
    
    return filtered


def _ndimage_filter_extreme_durations(da, min_duration=5, max_gap=2, n_jobs=36):
    """
    Allow only extreme events that meet a minimum duration requirement, allowing
    for gaps. 
    
    Uses binary closing to bridge gaps and then binary opening to drop events 
    that do not meet the minimum duration requirement. The default configuration 
    is set to match the set-up from Hobday et al. (2018).
    
    Parameters
    ----------
    da : xr.DataArray
        the data array where `time` must be one of the dimensions. The function
        will transpose if time is not the first dimension. 
    min_duration : int [5]
        the minimum duration of an event (days/months). Events shorter than this 
        duration will be dropped. Any even numbers will be rounded up so that
        there is a middle value. 
    max_gap : int [3]
        the maximum duration of a gap between events so that the event is 
        considered one event (in units of the time step).
    n_jobs : int [36]
        the number of jobs that the process will be run on (note that this only
        works when run with _run_blocks)
        
    Returns
    -------
    xr.DataArray 
        has exactly the same dims and coords as the input array, with some 
        changes to the filtering steps. 
        
    Note
    ----
    This method is about three times faster than using run length encoding (a 
    different approach to get the same result). However, this is only true when
    the min_duration and max_gap are ≲ 20. Beyond this, run length encoding 
    should be used. Further, run length encoding is easier to parallelise since
    the approach runs only along the time dimension.
    """
    from scipy import ndimage
    import numpy as np
    
    def make_time_dim_structure(n, ndim):
        """creates a convolution structure where only the center value of the 
        space dimension is 1, meaning that convolution runs along the time 
        dimension. The rest is set to zero.
        """
        m = n // 2 * 2 + 1
        i = m // 2
        structure = np.zeros([m] * ndim, dtype=int)
        structure[(slice(None),) + (slice(i, None, m),) * (ndim - 1)] = 1
        return structure
    
    dims = list(da.dims)
    dims.remove('time')
    da = da.transpose('time', *dims)
    arr = da.values
    
    ndim = arr.ndim
    
    # adding 1 to max gap makes the allowable gap work as expected. This was
    # checked against the run length encoding method. 
    closing_structure = make_time_dim_structure(max_gap + 1, ndim)
    opening_structure = make_time_dim_structure(min_duration, ndim)
    
    closed = ndimage.binary_closing(arr, structure=closing_structure, )
    opened = ndimage.binary_opening(closed, structure=opening_structure)
    
    out = xr.DataArray(opened, dims=da.dims, coords=da.coords)
    
    return out


def _run_blocks(da, func, n_jobs=36, chunks=None, **kwargs):
    from dask.diagnostics import ProgressBar
    
    if chunks is None:
        # chunk lat and lon so that the number of chunks is roughly n_jobs * 2
        c = ((2 * (n_jobs - 1))**0.5) // 1
        y = int(da.lat.size // c)
        x = int(da.lon.size // c)
        chunks = {'time': -1, 'lat': y, 'lon': x}
        
    with ProgressBar():
        print(f'Rechunking data for efficient parallel processing ({str(chunks)})')
        da = da.chunk(chunks)
        print(f'Filtering event durations ({str(kwargs)})')
        filtered = da.map_blocks(
            func, 
            kwargs=kwargs,
            template=da,
        ).compute(num_workers=n_jobs)
        
    return filtered
    
    
@wraps(_rle_filter_extreme_durations)
def filter_extreme_event_duration_runlen(mask, min_duration=5, max_gap=2, n_jobs=36, chunks=None):
    return _run_blocks(
        mask, 
        _rle_filter_extreme_durations, 
        n_jobs=n_jobs, 
        chunks=chunks,
        max_gap=max_gap, 
        min_duration=min_duration)


@wraps(_ndimage_filter_extreme_durations)
def filter_extreme_event_duration_ndimage(mask, min_duration=5, max_gap=2, n_jobs=36, chunks=None):
    return _run_blocks(
        mask, 
        _ndimage_filter_extreme_durations, 
        n_jobs=n_jobs, 
        chunks=chunks,
        max_gap=max_gap, 
        min_duration=min_duration)

