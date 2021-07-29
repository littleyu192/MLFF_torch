all: gen_3b_feature.x gen_2b_feature.x  write_egroup.x
	\cp gen_3b_feature.x gen_2b_feature.x  write_egroup.x ../../bin/

gen_3b_feature.x: 
	ifort scan_title.f90 transform_to_upper.f90 get_grid3b_type1.f90 get_grid3b_type2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_3b_type1.f90 \
        find_feature_3b_type2.f90 find_feature_3b_type3.f90 gen_3b_feature.f90 \
		-o $@
gen_2b_feature.x: 
	ifort scan_title.f90 transform_to_upper.f90 get_grid2b_type1.f90 get_grid2b_type2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_2b_type1.f90 \
        find_feature_2b_type2.f90 find_feature_2b_type3.f90 gen_2b_feature.f90 \
		-o $@
write_egroup.x:
	ifort scan_title.f90 transform_to_upper.f90  write_egroup.f90 \
        find_neighbore00.f90 find_neighbore.f90  \
		-o $@
clean:
	rm -f *.x *.mod
