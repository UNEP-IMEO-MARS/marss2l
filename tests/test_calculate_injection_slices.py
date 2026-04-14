import unittest
from marss2l.mars_sentinel2.plumesimulation import calculate_injection_slices


class TestCalculateInjectionSlices(unittest.TestCase):
    def assert_slices(self, image_slice, plume_slice, image_shape):
        self.assertEqual(
            image_slice[0].stop - image_slice[0].start,
            plume_slice[0].stop - plume_slice[0].start,
        )
        self.assertEqual(
            image_slice[1].stop - image_slice[1].start,
            plume_slice[1].stop - plume_slice[1].start,
        )
        self.assertTrue(0 <= image_slice[0].start < image_shape[0])
        self.assertTrue(0 <= image_slice[0].stop <= image_shape[0])
        self.assertTrue(0 <= image_slice[1].start < image_shape[1])
        self.assertTrue(0 <= image_slice[1].stop <= image_shape[1])

    def test_within_bounds(self):
        image_shape = (100, 100)
        plume_shape = (20, 20)
        loc_injection = (50, 50)
        image_slice, plume_slice = calculate_injection_slices(
            image_shape, plume_shape, loc_injection
        )
        self.assert_slices(image_slice, plume_slice, image_shape)

    def test_outside_bounds_top_left(self):
        image_shape = (100, 100)
        plume_shape = (20, 20)
        loc_injection = (-10, -10)
        image_slice, plume_slice = calculate_injection_slices(
            image_shape, plume_shape, loc_injection
        )
        self.assert_slices(image_slice, plume_slice, image_shape)

    def test_outside_bounds_bottom_right(self):
        image_shape = (100, 100)
        plume_shape = (20, 20)
        loc_injection = (90, 90)
        image_slice, plume_slice = calculate_injection_slices(
            image_shape, plume_shape, loc_injection
        )
        self.assert_slices(image_slice, plume_slice, image_shape)

    def test_partial_overlap_top_left(self):
        image_shape = (100, 100)
        plume_shape = (20, 20)
        loc_injection = (-5, -5)
        image_slice, plume_slice = calculate_injection_slices(
            image_shape, plume_shape, loc_injection
        )
        self.assert_slices(image_slice, plume_slice, image_shape)

    def test_partial_overlap_bottom_right(self):
        image_shape = (100, 100)
        plume_shape = (20, 20)
        loc_injection = (95, 95)
        image_slice, plume_slice = calculate_injection_slices(
            image_shape, plume_shape, loc_injection
        )
        self.assert_slices(image_slice, plume_slice, image_shape)


if __name__ == "__main__":
    unittest.main()
