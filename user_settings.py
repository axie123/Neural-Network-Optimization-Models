# Written by Dr.Pascal Frederich and later modified.
# From the Aspuru-Guzik Group, University of Toronto.

import os
import yaml

if os.path.exists("template_settings.yml"):
    user_settings=yaml.load(open("template_settings.yml","r"))
    print(user_settings)
else:
    print('Doesn\'t open')
