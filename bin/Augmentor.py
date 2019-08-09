import Augmentor

p = Augmentor.Pipeline('AugmentorData/blank45')
p.greyscale(probability=1.0)
p.skew(probability=0.9, magnitude=0.4)
# p.random_erasing(0.1, rectangle_area=0.3)
p.sample(20000)
