"""Compatibility import for the independently audited boundary-center oracle."""
try:
    from .angular_boundary_centers import boundary_centers
except ImportError:
    from angular_boundary_centers import boundary_centers

__all__ = ['boundary_centers']
