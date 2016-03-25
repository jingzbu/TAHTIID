TAHTIID
======

TAHTIID is short for Threshold Approximation for Hoeffding's Test under I.I.D. Assumption. The main theoretical result that we
apply in this package is from [J. Unnikrishnan and D. Huang](http://lcav.epfl.ch/files/content/sites/lcav/files/people/jayakrishnan.unnikrishnan/TIT13submitted.pdf).


Usage
=====
Assuming having set `tahtiid_ROOT` path variable correctly, then type `./tahtiid -h` to get the following help message:
```
usage: tahtiid [-h] [-e E] [-beta BETA] [-N N] [-fig_dir FIG_DIR] [-show_pic]

optional arguments:
  -h, --help        show this help message and exit
  -e E              experiment type; indicated by 'eta' (threshold calculation
                    and visualization) or 'cdf' (empirical CDF calculation and
                    visualization); default='eta'
  -beta BETA        false alarm rate for Hoeffding's rule; default=0.001
  -N N              total number of labels of the i.i.d. random variables;
                    default=4
  -fig_dir FIG_DIR  folder for saving the output plot; default='./Results/'
  -show_pic         whether or not to show the output plot; default=False
```

Examples:

 `$ ./tahtiid -beta 0.1 -N 3`
 
 `$ ./tahtiid -e cdf -beta 0.01 -N 3 -show_pic`

 `$ ./tahtiid -beta 0.01 -N 4 -show_pic`

 `$ ./tahtiid -beta 0.0001 -N 5 -show_pic`



Author
=============
Jing Zhang

Jing Zhang currently is a PhD student in the Division of Systems Engineering at Boston University, advised by Professor [Yannis Paschalidis](http://sites.bu.edu/paschalidis/).


Email: `jzh@bu.edu`

Homepage: http://people.bu.edu/jzh/


Copyright 2015 Jing Zhang. All rights reserved. TAHTIID is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation.
