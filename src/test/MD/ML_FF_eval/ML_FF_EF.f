        subroutine MF_FF_EF(Etot,fatom,xatom,AL,natom)

        use mod_mpi
        use calc_feature
        use calc_lin
        implicit none

        integer natom
        real*8 Etot
        real*8 fatom(3,natom)
        real*8 xatom(3,natom)
        real*8 AL(3,3)


        call gen_feature(AL,xatom)
 
        call cal_energy_force(feat,dfeat,num_neigh,list_neigh,AL,xatom)

        Etot=Etot_pred
        fatom(:,1:natom)=force_pred(:,1:natom)

        return
        end subroutine MF_FF_EF
        

       

