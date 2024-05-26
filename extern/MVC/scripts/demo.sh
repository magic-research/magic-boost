export PYTHONPATH=$PYTHONPATH:./extern/MVC

# use instand3d
# python extern/MVC/scripts/demo.py --config_path extern/MVC/mvc/configs/mvc_instant3d.yaml \
#                                   --num_frames 5 \
#                                   --image_root load/mv_insatnt3d/armor \
#                                   --target_elevation 10. \
#                                   --target_azimuth 45. 

# use instantmesh
python extern/MVC/scripts/demo.py --config_path extern/MVC/mvc/configs/mvc_instantmesh.yaml \
                                  --num_frames 7 \
                                  --image_root load/mv_instantmesh/blue_cat \
                                  --target_elevation 10. \
                                  --target_azimuth 45. 