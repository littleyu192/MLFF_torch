subroutine input(AL,ilocal,TOLUG,tolE,niter0,niter1,nline0,nline1,iCGmth0,iCGmth1,iscfmth0,iscfmth1,FermidE0,FermidE1,itypeFermi0,itypeFermi1,mCGbad0,mCGbad1,islda,igga,TOTNEL,ntype,xatom,iatom,iiatom,ityatom,fxatom_out,iwg_in,fwg_in,iwg_out,fwg_out,irho_in,frho_in,irho_out,frho_out,fvr_in,ivr_out,fvr_out,iforce,fforce_out,ipsp_type,ipsp_all,vwr_atom,nkpt,smatr,nrot,num_mov,tolforce,imov_at,dtstart,dd_limit,idens_out,kpt_dens,ispin_dens,iw_dens,fdens_out,f_xatom,numref,nref_type,numref0,nref0_type, ivext_in,fvext_in,imv_cont,amx_mth0,amx_mth1,xgga,iMDatom,iMD,MDstep,dtMD,temperature1,temperature2,iscale_temp_VVMD,nstep_temp_VVMD,RHO_RELATIVE_ERROR,FORCE_RELATIVE_ERROR,iQijL0_GS,ISBF,tolRHO,istress_cal,tol_stress,fstress_out)

        !add a parameter xv_atom to store the position and velocity for MD simulation!!

        !******************************************
        !c     Written by Lin-Wang Wang, March 30, 2001.  
        !c     Copyright 2001 The Regents of the University of California
        !c     The United States government retains a royalty free license in this work
        !******************************************
        !c     
        !c     Gao Yan Tao. notes:
        !c     RDUM is a single precision number, please note, assign it to double 
        !c     precision might have potential error.
        !****************************************
        !****  It stores the wavefunction in G space, only in half
        !****  of the E_cut sphere. 
        !******************************************
        use vdw ! vdw
        use london ! vdw-london
        use dftd3_input_param
        use dftd3_api, only : dftd3_init, dftd3_set_params, dftd3_set_functional, dftd3_calc, dftd3_input
        use dftd3_qe, only : dftd3_printout, dftd3, dftd3_in

        use pseudo_potential ! upf
        use iotk_common_world  !zhilin
        use pwscf_control   !zhilin
        use xc  ! zhilin
        use xc_f90_lib_m ! zhilin
        use data_fft_L2_14
        use data_fft_L_13
        use data_fft_12
        use data_Gcolumn_wave_9
        use data_Gcolumn_L_10
        use data_Gcolumn_L2_11
        use data
        use common_module_99
        use param_escan
        use data_nonlocal_8
        use data_wave_7
        use data_pulay_6
        use data_vrho_5
        use data_vrho_G_3
        use data_commun_0
        use data_variable_1
        use gpu_module
        use hse_module
        use data_AL0
        use data_wkm
        use mod_control
        use pwmat2pwscf
        use mod_mpi
        use mod_smearing
        use mod_io
        use mod_relax
        use data_egghead
        use data_ext_force
        use solvent0_base
        use solvent0_init
        use fft_base, only : dfftp
        use atom_base
        use mod_tddft
        use mod_SCFEP
        use mod_CPMNLF
        use mod_energydecomp1
        use mod_energydecomp2
        use mod_plugin
        use mod_map_proj_occ
        use mod_md
        use mod_fixw
        use mod_NAMD
        use mod_Ef_spin
        use rVV10, only: iflag_rVV10,b_rVV10,c_rVV10

        use data_force_field, only : irmethod_vff, num_mov_vff, force_tolerance_vff, k_b, k_a, k_d, k_h, is_gupta

        use default_element, only : elements

        use mod_FSM_DOS
        use mod_magnetic
        use mod_neb
        use mod_ldau
        use mod_mixing
        use mod_jobhpsi
        use mod_rho_add
        use mod_rho_diag
        use mod_solvent, only: iflag_solvent_energydecomp,solvent_extr_eden
        use mod_trans
        use mod_SCF_NEQ

        implicit double precision (a-h,o-z)

        include 'mpif.h'
        !******************************************
        real*8 AL(3,3),AL_t(3,3),AL_grid(3,3)
        !***********************************************
        !****  single precision for vr(i)
        !***********************************************
        !*************************************************
        real*8 xatom(3,matom_1),iMDatom(matom_1),iMDtype(mtype_1)
        integer iatom(matom_1),imov_at(3,matom_1),iiatom(mtype_1),ityatom(matom_1)
        integer numref(matom_1),numref0(matom_1)
        integer smatr(3,3,48),nrot
        integer iCGmth0(mscf_1),iCGmth1(mscf_1),iscfmth0(mscf_1),iscfmth1(mscf_1)
        integer iCGmth0_,iCGmth1_,iscfmth0_,iscfmth1_
        real*8 amx_mth0(mscf_1),amx_mth1(mscf_1)
        real*8 amx_mth0_,amx_mth1_
        real*8 FermidE0(mscf_1),FermidE1(mscf_1),TOTNEL
        real*8 FermidE0_,FermidE1_
        integer itypeFermi0(mscf_1),itypeFermi1(mscf_1)
        integer itypeFermi0_,itypeFermi1_
        integer kpt_dens(2),ispin_dens(2),iw_dens(2)
        integer niter0,nline0,niter1,nline1,mCGbad0,mCGbad1
        integer niter0_,nline0_,niter1_,nline1_,mCGbad0_,mCGbad1_
        integer hse_nq1_t,hse_nq2_t,hse_nq3_t
        real*8 RHO_RELATIVE_ERROR, FORCE_RELATIVE_ERROR
        integer, allocatable, dimension(:) :: ind_tmp
        real*8, allocatable, dimension(:) :: weigh_tmp
        !integer lll_type(30,mtype_1),nbeta_type(mtype_1)

        !**************************************************
        character*3 psp_head
        integer ipsp_type(mtype_1),nref_type(mtype_1),nref0_type(mtype_1)
        real*8 z_type(mtype_1),Ecut_type(mtype_1),amass_type(mtype_1)
        character*200 vwr_atom(mtype_1),filename
        character*20 fforce_out,fdens_out
        character*100 fwg_in(2),frho_in(2)
        character*20 fwg_out(2),frho_out(2),fvr_in(2),fvr_out(2),f_tmp,fxatom_out,fvext_in
        character*20 f_xatom,sym_file,kpt_file
        character*20 xv_xatom
        character*200 message, tmp_char, mixing
        character*10 JOB, temp_job,CONVERGE, JOB_store
        character*8  ACCURACY
        !
        real*8,allocatable,dimension(:,:,:) :: xatom_inter
        ! for LDAU
        integer :: is_ldau_type_tmp,ldau_l_tmp
        real*8 :: Hubbard_U_tmp(2)
        !
        integer im2,im3,im1
        complex*16 cc
        real*8  sum


        integer :: igga
        real*8 :: xgga
        REAL       RDUM(20)
        INTEGER    IDUM(20)
        CHARACTER(60)  CHARAC
        INTEGER    N1_NUM(20) 
        COMPLEX    CDUM(20)
        LOGICAL    LDUM(20),FOUND,CONT,YINTST,LOPEN
        LOGICAL    L
        INTEGER    N,IERR
        character(len=200) temp_char
        character(len=11) precision_control
        character(len=256) temp_char_xc
        CHARACTER(LEN=200) :: right, temp_right
        LOGICAL :: right_logical, alive
        INTEGER :: flag_dens, i_plus
        !*************************************************

        integer :: istress_cal
        real*8  :: tol_stress
        character*20 fstress_out

        !! for split rcut by type
        real*8 max_rcut
        integer is_rcut_type_set
        integer is_rcut_set

        integer :: icgmth,iscfmth,itypeFermi
        real*8 :: fermidE
        integer :: i, j
        integer nk1,nk2,nk3,sk1,sk2,sk3,iflag_symm

        real(kind=8) :: HSE_omega_bohr

        real*8,allocatable, dimension (:,:) :: rhoq_tmp
        integer,allocatable, dimension (:) :: itmp

        real*8 :: N123L_scale_factor
        integer :: sys_type
        integer :: m1_td,m2_td,mstate_td
        ! solvent0
        integer, allocatable :: num11(:)
        character(len=2), allocatable :: element_name(:), element_name_all(:)
        real(kind=8), allocatable :: zv(:)
        integer, allocatable :: ityp_old(:)
        integer :: nnnstart, n1_init, n1_old
        ! dft-d3
        real(kind=8) :: pars(5)
        character(len=80) :: get_dft_short
        type(atom_config) :: ac
        !
        frho_in(1) = 'IN.RHO'
        frho_in(2)=  'IN.RHO_2'
        fwg_in(1) = 'IN.WG'
        fwg_in(2)=  'IN.WG_2'
        fvr_in(1) = 'IN.VR'
        fvr_in(2)=  'IN.VR_2'
        frho_in_SOM="IN.RHO_SOM";
        frho_out(1) = 'OUT.RHO'
        frho_out(2)=  'OUT.RHO_2'
        fwg_out(1) = 'OUT.WG'
        fwg_out(2)=  'OUT.WG_2'
        fvr_out(1) = 'OUT.VR'
        fvr_out(2)=  'OUT.VR_2'
        frho_out_SOM="OUT.RHO_SOM";

        message="   "

        if(inode_tot.eq.1) then
            open(22,file="REPORT")
            rewind(22)
        endif

        !------------------------------------------------------------------------
        open(9,file='etot.input',status='old',action='read',iostat=ierr) 

        if(ierr.ne.0) then
            message="file ***etot.input*** does not exist, stop"
            call error_stop(message,__FILE__,__LINE__)
        endif

        rewind(9)
        num_group_b = 1
        read(9,*) nnodes_b, num_group_k
        if(inode_tot.eq.1) then
            write(22,*) nnodes_b, num_group_k
        endif

        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        numMPIPerGPU = 1
        numMPIPerGPU_temp = 1;
        CALL read_key_words ( 9, "NUM_MPI_PER_GPU", LEN('NUM_MPI_PER_GPU'), right, IERR )
        if(ierr.eq.0) then
          READ ( right, * ) numMPIPerGPU_temp
          if( (numMPIPerGPU_temp.gt.32) .or.(numMPIPerGPU_temp .lt. 1))then
            numMPIPerGPU_temp = 1
          endif
        endif
        numMPIPerGPU = numMPIPerGPU_temp 

        !------------------------------------------------------------------------

        CALL read_key_words ( 9, "PRECISION", LEN('PRECISION'), right, IERR )
        if(ierr .eq. 0) then
            READ( right, "(A10)" ) precision_control
            call transform_to_upper(precision_control)
            precision_control = adjustl(trim(precision_control))
            if(trim(precision_control) .eq. "DOUBLE") then
                precision_flag = double_prec
            else if(trim(precision_control) .eq. "MIX") then
                precision_flag = mix_prec
            else if(trim(precision_control) .eq. "SINGLE") then
                precision_flag = single_prec
            else 
                precision_flag = auto_prec
            endif
        else
            PRECISION_CONTROL = "AUTO"
            precision_flag = auto_prec
        endif

        if(inode_tot.eq.1) then
            write(22,*) "PRECISION  = ", PRECISION_CONTROL
        endif

        !------------------------------------------------------------------------
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        iflag_wkm=0
        CALL read_key_words ( 9, 'JOB', LEN('JOB'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, "(A10)" ) JOB
            call transform_to_upper(JOB)
            if(trim(JOB).ne."SCF".and.trim(JOB).ne."NONSCF".and.trim(JOB).ne."DOS".and.trim(JOB).ne."RELAX".and. &
               trim(JOB).ne."MD".and.trim(JOB).ne."NEB".and.trim(JOB).ne."TEST".and.trim(JOB).ne."WKM".and. &
               trim(JOB).ne."EGGFIT".and.trim(JOB).ne. "TDDFT".and.trim(JOB).ne."NAMD" .and.trim(JOB).ne."SCFEP".and. &
               trim(JOB).ne."POTENTIAL".and.trim(JOB).ne."HPSI".and.trim(JOB).ne."WKMRELAX".and. &
               trim(JOB).ne."CPMNLF".and.trim(JOB).ne."MOMENT".and.trim(JOB).ne."TRANS") then

                message="JOB needs to be:SCF,NONSCF,DOS,MOMENT,SCFEP,RELAX,MD,NEB,EGGFIT,TDDFT,TEST,NAMD,POTENTIAL,HPSI,WKMRELAX,CPMNLF,TRANS"

                call error_stop(message,__FILE__,__LINE__)
            endif
        else
            message="JOB needs to be:SCF,NONSCF,DOS,MOMENT,SCFEP,RELAX,MD,NEB,EGGFIT,TDDFT,TEST,NAMD,POTENTIAL,HPSI,WKMRELAX,CPMNLF,TRANS"
            call error_stop(message,__FILE__,__LINE__)
        endif

        JOB_store=trim(JOB)
        if(trim(JOB).eq."POTENTIAL") JOB="SCF"
        JOB_1 =  trim(JOB)

        iflag_CPMNLF=0
        if(trim(JOB).eq."CPMNLF")  then
        JOB="SCF"
        JOB_1 =  trim(JOB)
        iflag_CPMNLF=1
           CALL read_key_words ( 9, 'CPMNLF_DETAIL', LEN('CPMNLF_DETAIL'),right, IERR )
           if(ierr.eq.0) then
                READ (right,*,IOSTAT=ierror) isi_CPMNLF,isf_CPMNLF,kpt_CPMNLF,iislda_CPMNLF
                if ( inode_tot == 1 ) then
                 write (22, 2019) isi_CPMNLF,isf_CPMNLF,kpt_CPMNLF,iislda_CPMNLF
                    2019         format("CPMNL_DETAIL = ", 1x, 4(i7,1x))
                endif
            else
                message="for JOB=SCFEP, need SCFEP_DETAIL line"
                call error_stop(message,__FILE__,__LINE__)
            endif


        endif

        if(inode_tot.eq.1) then
            write(22,*) "JOB       = ", trim(JOB_store)
        endif
        if(JOB_1.eq."WKM") then
            iflag_wkm=1
            JOB="SCF"
            JOB_1 = trim(JOB)
        endif
        if(JOB_1.eq."WKMRELAX") then
            iflag_wkm=1
            JOB="RELAX"
            JOB_1 = trim(JOB)
        endif

        iflag_hpsi=0
        if(JOB_1.eq."HPSI") then
          iflag_hpsi=1
          JOB="NONSCF"
          JOB_1=trim(JOB)
        endif

        !       if(trim(JOB) .eq. "TEST") JOB = "RELAX"   ! relax if job equals TEST ???
        if(trim(JOB) .eq. "TEST1") JOB = "RELAX"  ! for future feature.
        if(trim(JOB) .eq. "TEST2") JOB = "RELAX"  ! for future feature.
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if(trim(JOB).eq."DOS") then
            ido_DOS_999=1
        else
            ido_DOS_999=0
        endif

        if(trim(JOB).eq."MOMENT") then
            ido_moment_999=1
        else
            ido_moment_999=0
        endif
        if(trim(JOB).eq."TRANS") then
            JOB="NONSCF"
            JOB_1=trim(JOB)
            iflag_trans=1
        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        !  calculated the species of the elements ( ntype )
        ntype = 0
        DO i = 1, 100
            !temp_char = "IN.PSP"//CHAR(i+48)
            write(temp_char,*) i
            temp_char="IN.PSP"//ADJUSTL(trim(temp_char))
            temp_char = ADJUSTL(temp_char)
            CALL read_key_words ( 9, TRIM(temp_char),len_trim(temp_char), right, IERR )
            IF( IERR == 0 ) THEN
                ntype = ntype + 1
            ELSE
                EXIT
            ENDIF
            IF(i .gt. mtype_1) then
                message="Too much atom types. The upper limit is 10!"
                call error_stop(message,__FILE__,__LINE__)
            ENDIF
        ENDDO
        ntype_123=ntype

        allocate ( upfpsp(ntype) )  ! upf
        do I = 1,ntype
            !temp_char = "IN.PSP"//char(48+I)
            write(temp_char,*) i
            temp_char="IN.PSP"//ADJUSTL(trim(temp_char))
            temp_char = ADJUSTL(temp_char)
            CALL read_key_words ( 9, TRIM(temp_char),len_trim(temp_char), right, IERR )
            if(ierr.ne.0) then
                message="Need IN.PSP# for NUM_PSP atom"
                call error_stop(message,__FILE__,__LINE__)
            endif
            !vwr_atom(I) = CHARAC
            READ ( right, "(A200)" ) vwr_atom(i)
            inquire ( file = vwr_atom(i), exist = alive )
            if ( .not. alive ) then
                message='"'//trim(vwr_atom(i))//'"'//' does not exist!'
                call error_stop(message,__FILE__,__LINE__)
            end if
            !
            !c           psp_head = "NUL"
            !c           call trim_string(vwr_atom(i), psp_head)
            !c           if(psp_head.eq."usp") then          ! NEED TO BE FIXED LATER
            !c               ipsp_type(i)=2
            !c           elseif (psp_head .eq."vwr") then
            !c               ipsp_type(i)=1
            !c           else
            !c           if (index(vwr_atom(i),"usp") >0 ) then
            !c               ipsp_type(i) = 2
            !c           else
            !c               if (index(vwr_atom(i),"vwr") > 0) then
            !c                   ipsp_type(i) = 1
            !c               else
            !c                   message =
            !c     &         "IN.PSP# must be vwr.xxx(norm),or uspp.xxx(ultra-soft)"
            !c               call error_stop(message,__FILE__,__LINE__)
            !c                 endif
            !c           endif
            call read_upf_v2 (vwr_atom(i), upfpsp(i),ido_DOS_999 ) ! upf
            if ( upfpsp(i)%tvanp ) then
                ipsp_type(i) = 2
            else 
                ipsp_type(i) = 1
            end if
            if(inode_tot.eq.1) then
                write(22,20171) TRIM(temp_char),trim(vwr_atom(I))
                20171      format(1x,a,"   = ",a)
#ifdef TIMING
                write ( 6, * ) "ipsp_tyep, i", i, ipsp_type(i)
#endif
            endif
        enddo
        !--------------------------------------------------------------------------
          CALL read_key_words ( 9,'ENERGY_DECOMP_SPECIAL',LEN('ENERGY_DECOMP_SPECIAL'),right, IERR) 
          if(ierr.eq.0) then
            read(right,*) (weight_atom_energy_decomp(i),i=1,ntype)
           else
             do i=1,ntype
             weight_atom_energy_decomp(i)=1.d0
             enddo
          endif

          CALL read_key_words ( 9,'ENERGY_DECOMP_SPECIAL1',LEN('ENERGY_DECOMP_SPECIAL1'),right, IERR) 
          if(ierr.eq.0) then
            READ(right,*) (eta_fact(ii),ii=1,ntype)
          else
              eta_fact(1:ntype)=1.d0
          endif

          CALL read_key_words ( 9,'ENERGY_DECOMP_SPECIAL2',LEN('ENERGY_DECOMP_SPECIAL2'),right, IERR) 
          if(ierr.eq.0) then
            read(right,*) energy_decomp_exp
           else
             energy_decomp_exp=1.d0
          endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'IN.ATOM',LEN('IN.ATOM'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, "(A20)" ) f_xatom
        else
            f_xatom = "atom.config"
        endif

        !just read IMD, if IMD=11 when JOB=TDDFT, f_xatom=TDDOS/restart.config
        if(trim(JOB).eq."TDDFT") then
            CALL read_key_words(9,'MD_DETAIL',LEN('MD_DETAIL'),right,IERR)
            if(ierr.eq.0) then
                READ ( right, * ) iMD
            else
                message="For JOB .eq. MD .or. TDDFT, must have MD_DETAIL with MDmethod, MDstep, dtMD, Temperature1, Temperature2"
                call error_stop(message,__FILE__,__LINE__)
            endif
            if(iMD.eq.11) then
                f_xatom='TDDOS/restart.config'
            endif
        endif

        if(inode_tot.eq.1) then
            write(22,*) "IN.ATOM   = ", f_xatom
        endif
        call readf_xatom_new()
        !!readf_xatom_new() can be replace by following lines
        !call read_atom(f_xatom,upfpsp,ac)
        !call write_atom('test.config',ac)
        !call old_fashion_xatom_assign(ac)
        !call deallocate_atom_config(ac)
        !
        do ia  = 1, ntype
            iiatom(ia) = upfpsp(ia)%num
        enddo
        call init_atom_base(AL,natom,ntype,xatom,imov_at,iatom,iiatom)
        allocate (ityp(sum_atom))
        do i = 1, sum_atom
            jloop: do j = 1, sum_atom_type
                if (atom(i)%number == upfpsp(j)%num) then
                    ityp(i) = j
                    exit jloop
                end if
            end do jloop
        end do
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        !     Weile Jia, add the easy/difficult convergence.
        CALL read_key_words(9,'CONVERGENCE',LEN('CONVERGENCE'),right,IERR)
        IF(IERR .eq. 0) then
            call transform_to_upper(right)
            read (right, '(A10)')  CONVERGE
        ELSE
            CONVERGE = "EASY"
        ENDIF
        if(CONVERGE.ne."EASY" .and. CONVERGE.ne."DIFFICULT") then
            if(inode_tot.eq.1) then
                write(*,*) "CONVERGE  is set to a wrong value."
                write(*,*) "change it to default value 'EASY'."
            endif
            CONVERGE = "DIFFICULT"
        endif
        if(inode_tot.eq.1) then
            write(22,*) "CONVERGENCE  = ", trim(CONVERGE)
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !      Weile Jia, add the high/low accuracy.
        CALL read_key_words(9,'ACCURACY',LEN('ACCURACY'),right,IERR)
        IF(IERR .eq. 0) then
            call transform_to_upper(right)
            read (right, '(A8)')  ACCURACY
        ELSE
            ACCURACY = "NORM"
            !if(trim(JOB).eq."RELAX" .or. trim(JOB) .eq. "NEB") ACCURACY="HIGH"
        ENDIF
        if(ACCURACY .ne. "HIGH" .and. ACCURACY .ne. "NORM" .and. ACCURACY .ne. "VERYHIGH") then
            if(inode_tot.eq.1) then
                write(*,*) "ACCURACY  is set to a wrong value."
                write(*,*) "change it to default value 'NORM'."
            endif
            ACCURACY = "NORM"
        endif
        !!  set default
        !if(CONVERGE .eq. "DIFFICULT") then
        !     ACCURACY = "HIGH"
        !endif 
        !
        if(inode_tot.eq.1) then
            write(22,*) "ACCURACY  = ", trim(ACCURACY)
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        itype_interp=1
        if(trim(JOB).eq."DOS") then
            CALL read_key_words ( 9, 'DOS_DETAIL', LEN('DOS_DETAIL'),right, IERR )
            if(ierr.eq.0) then
                READ (right,*,IOSTAT=ierror)interp_DOS_999,hse_nq1_t,hse_nq2_t,hse_nq3_t,ipart_DOS 
                if(ierror.ne.0) then
                    READ (right,*,IOSTAT=ierror)interp_DOS_999,hse_nq1_t,hse_nq2_t,hse_nq3_t
                    ipart_DOS=0
                endif
            if(inode_tot.eq.1) then
            WRITE (22, 2015) interp_DOS_999,hse_nq1_t,hse_nq2_t,hse_nq3_t,ipart_DOS
            2015    format(" DOS_DETAIL    = ",i1,3x,4(i6,2x))
            endif
                if(interp_DOS_999.eq.1) then
                itype_interp=1
                elseif(interp_DOS_999.eq.2) then
                itype_interp=2
                interp_DOS_999=1 ! be care
                endif
            else
                interp_DOS_999=0
                hse_nq1_t=1
                hse_nq2_t=1
                hse_nq3_t=1
                ipart_DOS=0
            endif
        else
            interp_DOS_999=0
            hse_nq1_t=1
            hse_nq2_t=1
            hse_nq3_t=1
            ipart_DOS=0
        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc          
        iflag_SCFEP=0
        if(trim(JOB).eq."SCFEP") then
            iflag_SCFEP=1
            CALL read_key_words ( 9, 'SCFEP_DETAIL', LEN('SCFEP_DETAIL'),right, IERR )
            if(ierr.eq.0) then
                READ (right,*,IOSTAT=ierror) is1_SCFEP,is2_SCFEP,alpha_SCFEP,kpt_SCFEP,iislda_SCFEP
                if ( inode_tot == 1 ) then
                    write (22, 2018) is1_SCFEP,is2_SCFEP,alpha_SCFEP,kpt_SCFEP,iislda_SCFEP

                    2018     format("SCFEP_DETAIL = ", 1x, 2(i7,1x),1x,F15.8,2x,i5,1x,i2)
                endif
            else
                message="for JOB=SCFEP, need SCFEP_DETAIL line"
                call error_stop(message,__FILE__,__LINE__)
            endif

            CALL read_key_words ( 9, 'IN.FORCE', LEN('IN.FORCE'),right, IERR )
            if(ierr.eq.0) then
                READ (right,*,IOSTAT=ierror) right_logical 
                if ( right_logical ) then
                    iflag_inforce=1
                else
                    iflag_inforce=0
                endif
            else
                iflag_inforce=0
            endif
            if(iflag_inforce==0) then
                if(inode_tot .eq. 1) then
                    write(6,*) "**********************************************************"
                    write(6,*) "WARNING: FILE IN.FORCE NOT READIN"
                    write(6,*) "**********************************************************"
                    write(22,*) "**********************************************************"
                    write(22,*) "WARNING: FILE IN.FORCE NOT READIN"
                    write(22,*) "**********************************************************"
                endif
            endif
        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc          
        tol_corr=-1.d-3
        if(trim(JOB).eq."RELAX") then
            dtstart = 1.d0
            imv_cont = 0
            CALL read_key_words ( 9, 'RELAX_DETAIL', LEN('RELAX_DETAIL'),right, IERR )
            if(ierr.eq.0) then
                READ (right,*,IOSTAT=ierror) irmethod_1,num_mov,tolforce,istress_cal,tol_stress,tol_corr,dt_max,dL_max
                if(ierror .ne. 0) then
                    dt_max=0.1
                    dL_max=0.1
                    READ (right,*,IOSTAT=ierror) irmethod_1,num_mov,tolforce,istress_cal,tol_stress,tol_corr
                endif
                if(ierror .ne. 0) then
                    READ (right,*,IOSTAT=ierror) irmethod_1,num_mov,tolforce,istress_cal,tol_stress
                endif
                if(ierror .ne. 0) then
                    READ (right,*) irmethod_1,num_mov,tolforce
                    istress_cal=0
                    tol_stress=0.d0
                endif
                tolforce = tolforce 
                num_mov = num_mov
                imv_cont = imv_cont
                fxatom_out = "xatom.relax"
            else
                !ccccccc default values for atomic relaxation
                !write(*,*) 'use default....'
                irmethod_1 = 1
                num_mov    = 200
                if(ACCURACY.eq.'NORM') then
                    tolforce   = 0.02    
                else
                    tolforce   = 0.01     ! for large system, this is what it can achieve, can be ~ 1 meV, good enough
                endif
                temp_right = ''
                call read_key_words (9, 'XCFUNCTIONAL', len('XCFUNCTIONAL'), right, ierr)
                temp_right = trim(adjustl(right))
                call transform_to_upper (temp_right)
                if (index((temp_right),'HSE')>0) tolforce = 0.03
                !if(use_hse .eq. 1) tolforce=0.03
                istress_cal = 0
                tol_stress = 0.d0
                fxatom_out = "xatom.relax"
            endif
            if(inode_tot.eq.1) then
                write(temp_char,2005)irmethod_1,num_mov,tolforce,istress_cal,tol_stress,tol_corr
                WRITE (22, *) TRIM(ADJUSTL(temp_char))
            endif
            tolforce = tolforce / Hartree_ev  * A_AU_1
            tol_stress = tol_stress  / Hartree_ev
            !write(*,*) tolforce ,'tolforce.................'
            2005    format("RELAX_DETAIL = ",i1,1x,i5,1x,(E13.5,2x),1x,i1,1x,(E13.5,2x),1x,(E13.5))
        else
            num_mov = 0
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if (irmethod_1 == 4) then
            dtstart_vff = 1.0d0
            call read_key_words ( 9, 'VFF_DETAIL', len('VFF_DETAIL'), right, ierr)
            if ( ierr == 0 ) then
                read (right, *, iostat = ierr) irmethod_vff, num_mov_vff, force_tolerance_vff, k_b, k_a, k_d, k_h
                if (ierr /= 0 .and. inode_tot == 1) then
                    write (6, *) "something wrong in reading VFF_DETAIL"
                    call mpi_abort(mpi_comm_world,ierr)
                end if
            end if
        end if
        if(inode_tot == 1) then
            write (tmp_char, "(i3,x,i5,x,E15.8,x,4f12.5)") irmethod_vff, num_mov_vff, force_tolerance_vff, k_b, k_a, k_d, k_h
            write(22,"(a,$)") ' VFF_DETAIL = '//trim(tmp_char)
            write (22, *) " "
        end if

        if(trim(JOB).eq."EGGFIT") then
            CALL read_key_words ( 9, 'EGG_DETAIL', LEN('EGG_DETAIL'),right, IERR )
            if(ierr.eq.0) then
                READ (right,*,IOSTAT=ierror) np1_egg,np2_egg,np3_egg
            else
                np1_egg=3
                np2_egg=3
                np3_egg=3
            endif
        else
            np1_egg=3
            np2_egg=3
            np3_egg=3
        endif
        if (inode_tot == 1) then
            write (tmp_char, *) np1_egg, np2_egg, np3_egg
            write(22,*) 'EGG_DETAIL = '//trim(tmp_char)
        end if

        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if(trim(JOB).eq."NEB") then
            CALL read_key_words ( 9, 'NEB_DETAIL', LEN('NEB_DETAIL'),right, IERR )
            dtstart = 1.d0
            imv_cont = 0
            if(ierr.eq.0) then
                READ (right,*) irmethod_neb_1,num_mov,tolforce,Nimage,ak_string_1,itype_string_1,Etotneb1_1,Etotneb2_1,itype_xatom, f_xatom
                if ( inode_tot == 1 ) then
                    write (22, 1155) irmethod_neb_1,num_mov,tolforce,Nimage,ak_string_1,itype_string_1,Etotneb1_1,Etotneb2_1,itype_xatom, f_xatom
                endif
                tolforce = tolforce 
                num_mov = num_mov
                imv_cont = imv_cont
                Etotneb1_1=Etotneb1_1/Hartree_ev
                Etotneb2_1=Etotneb2_1/Hartree_ev
                ak_string_1=ak_string_1/Hartree_eV* A_AU_1
            else
                message="for JOB=NEB, need NEB_DETAIL line"
                call error_stop(message,__FILE__,__LINE__)
            endif
            tolforce = tolforce / Hartree_ev  * A_AU_1

            if(irmethod_neb_1.ne.3.and.inode_tot.eq.1) then
                !    write(6,*) " WARNING: irmethod.ne.3 for NEB, might not converge!!"
                !    write(6,*) " WARNING: you might want to use irmethod.eq.3"
                !    write(22,*) " WARNING: irmethod.ne.3 for NEB, might not converge!!"
                !    write(22,*) " WARNING: you might want to use irmethod.eq.3"
            endif


            allocate(xatom_all_1(3,natom,0:Nimage+1))
            !
            call allocate_mod_neb()
            call read_key_words ( 9, 'IN.ATOM',LEN('IN.ATOM'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, "(A20)" ) neb_atomfile_ini
            else
                messgae="wrong in.atom"
                call error_stop(message,__FILE__,__LINE__)
            endif
            neb_atomfile_fin=f_xatom
            !
            if(itype_xatom.eq.1) then
                call read_atom(neb_atomfile_ini,upfpsp,neb_ac_all_image(0))
                call read_atom(neb_atomfile_fin,upfpsp,neb_ac_all_image(Nimage+1))
                call split_ac(neb_ac_all_image,Nimage)
                !
               ! do i=1,natom
               !     xatom_all_1(:,i,0)=xatom(:,i)
               ! enddo

               ! call readf_xatom_new()

               ! do i=1,natom
               !     xatom_all_1(:,i,Nimage+1)=xatom(:,i)
               ! enddo
               ! !
               ! call split_xatom()   ! generate the Nimage images, use straigt line
            else if(itype_xatom.eq.2) then
                !call readf_xatom_multi(Nimage)
                call read_multi_atom(f_xatom,upfpsp,neb_ac_all_image,Nimage)
                call mpi_barrier(mpi_comm_world,ierr)
!            else if(itype_xatom.eq.3) then
!                call readf_xatom_multi(Nimage)
!                interNimage=1
!                if(interNimage.gt.0) then
!                    allocate(xatom_inter(3,natom,0:(Nimage+1)*interNimage+Nimage+1))
!                    do i=1,Nimage+1
!                        call split_xatom_inter(xatom_all_1(1,1,i-1),xatom_all_1(1,1,i),xatom_inter(1,1,(i-1)*(interNimage+1)+1))
!                    enddo
!                    xatom_inter(:,:,0)=xatom_all_1(:,:,0)
!                    deallocate(xatom_all_1)
!                    allocate(xatom_all_1(3,natom,0:(Nimage+1)*interNimage+Nimage+1))
!                    xatom_all_1=xatom_inter
!                    Nimage=(Nimage+1)*interNimage+Nimage
!                    deallocate(xatom_inter)
!                endif
            else
                !write(message,*) "itype xatom should be 1,2,3, stop"
                write(message,*) "itype xatom should be 1,2, stop"
                call error_stop(message,__FILE__,__LINE__)
            endif



            call system('mkdir -p tmp_dump')

            1155   format(" NEB_DETAIL =",i3,x,i5,x,E12.5,x,i3,x,E12.5,x,i3,x,E20.10,1x,E20.10,x,i3,x,A)

        endif
        if(JOB=="NEB".or.JOB=='RELAX') then
            CALL read_key_words ( 9, 'IN.RELAXOPT', LEN('IN.RELAXOPT'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, * ) right_logical 
                if(right_logical) then
                    call read_relaxopt()
                else
                    call default_relaxopt()
                endif
                if(inode_tot.eq.1) then
                    write(22,*) "IN.RELAXOPT    = ", right_logical
                endif
            else
                call default_relaxopt()
                if(inode_tot.eq.1) then
                    write(22,*) "IN.RELAXOPT    = ", .false.
                endif
            endif
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        allocate (rcut_of_type(ntype)) ! diff ruct for diff type
        allocate (cu_wrmask_type(natom))
        allocate (cu_wrmaskxx_type(natom))
        allocate (cu_wrmaskxy_type(natom))
        allocate (cu_wrmaskxz_type(natom))

        allocate (cu_wrmask11_type(natom))
        allocate (cu_wrmask12_type(natom))
        allocate (cu_wrmask13_type(natom))
        allocate (cu_wrmask21_type(natom))
        allocate (cu_wrmask22_type(natom))
        allocate (cu_wrmask23_type(natom))
        allocate (cu_wrmask31_type(natom))
        allocate (cu_wrmask32_type(natom))
        allocate (cu_wrmask33_type(natom))

        allocate (single_cu_wrmask_type(natom))
        allocate (single_cu_wrmaskxx_type(natom))
        allocate (single_cu_wrmaskxy_type(natom))
        allocate (single_cu_wrmaskxz_type(natom))

        allocate (single_cu_wrmask11_type(natom))
        allocate (single_cu_wrmask12_type(natom))
        allocate (single_cu_wrmask13_type(natom))
        allocate (single_cu_wrmask21_type(natom))
        allocate (single_cu_wrmask22_type(natom))
        allocate (single_cu_wrmask23_type(natom))
        allocate (single_cu_wrmask31_type(natom))
        allocate (single_cu_wrmask32_type(natom))
        allocate (single_cu_wrmask33_type(natom))
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        iflag_fixw=0
        CALL read_key_words ( 9, "FIXW", LEN('FIXW'), right, IERR )
        if(ierr.eq.0) then
            read ( right, * ) right_logical
            if ( right_logical ) then
                iflag_fixw=1
            else
                iflag_fixw=0
            endif
            if(inode_tot.eq.1) write(22,*) "FIXW  =", right_logical
        else
            iflag_fixw=0
        endif



        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, "SPIN", LEN('SPIN'), right, IERR )
        ispin_input=1          
        if(ierr.eq.0) then
            READ ( right, * ) islda
            ispin_input=islda
            if(islda.eq.22.or.islda.eq.222) then
                is_SO=1
                !    if(precision_flag.ne.single_prec) then
                !   write(6,*) "SPIN=22,222(SO),can only use single precison,stop"
                !    stop
                !    endif
                if(islda.eq.22) then
                    is_SOM=0
                else
                    is_SOM=1        ! noncollinear magnetic field, will have rho_n_rho  
                endif

                islda=1
            else
                is_SO=0
                is_SOM=0
            endif
        else
            islda=1
            is_SO=0
        endif
        if(iflag_wkm.eq.1.and.islda.ne.2) then
            write(message,*) "for WKM calculation, spin must be 2"
            call error_stop(message,__FILE__,__LINE__)
        endif
        1114   format(" SPIN      = ",i3)
        !
        if(islda.ne.1.and.islda.ne.2.and.inode.eq.1) then
            write(message,*) "SPIN  must be 1 (lda) or 2 (slda), stop", islda
            call error_stop(message,__FILE__,__LINE__)
        endif

        if(is_SOM.eq.1) then
            CALL read_key_words ( 9, "SOM_SPHERE_RCUT", LEN('SOM_SPHERE_RCUT'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, * ) SOM_sphere_rcut
            else
                SOM_sphere_rcut=1.d+10 
            endif
            if (inode_tot.eq.1) write(22,*) "SOM_SPHERE_RCUT= ",SOM_sphere_rcut 
        endif
        if(inode_tot.eq.1) write(22,1114) ispin_input

        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        iconstraint_mag=0
        rcut_mag=5.d0
        allocate(amag_atom(natom))
        amag_atom=0.d0
            CALL read_key_words ( 9, "CONSTRAINT_MAG", LEN('CONSTRAINT_MAG'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, *,iostat=ierror ) iconstraint_mag
                if(ierror.ne.0) then
                 write(6,*) "CONSTAINT_MAG input line wrong,stop"
                 stop
               endif
            else
                iconstraint_mag=0
            endif
            if (inode_tot.eq.1) write(22,*) "CONSTRAINT_MAG =",iconstraint_mag

        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        ispin_wkm=100  ! no longer used
        nblock_band_fix=0
        band_exclude_Hxc=0

        if(iflag_WKM.eq.1) then
        call read_IN_S_WKM()
        endif


        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        ipsp_all=1
        do ia=1,ntype
            if(ipsp_type(ia).eq.2) ipsp_all=2
        enddo

        if(ilocal.eq.1) ipsp_all=1       ! bug
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !c       do ia=1,ntype
        !c           if(ipsp_type(ia).eq.1) then
        !c               call readvwr_head()
        !c           else
        !c              call readusp_head(vwr_atom(ia),iiatom(ia),nref_type(ia),
        !c     &    Z_type(ia),Ecut_type(ia),amass_type(ia),ido_DOS_999,
        !c     &    lll,nebta)
        !c           endif
        !c       enddo
        do ia  = 1, ntype
            iiatom(ia) = upfpsp(ia)%num
#ifdef TIMING
            write ( 6, * ) "iiatom(ia)", ia, iiatom(ia)
#endif
            nref_type(ia) = 0
            nref0_type(ia) = 0
            if(ido_DOS_999.eq.0) then


                num_tmp=0
                do j = 1, upfpsp(ia)%nbeta
                    if(is_SO.eq.1) then
                        nref_type(ia) = nref_type(ia) + upfpsp(ia)%lll(j)*2 +1
                    elseif(upfpsp(ia)%lllso(j).eq.0)  then
                        nref_type(ia) = nref_type(ia) + upfpsp(ia)%lll(j)*2 +1
                    endif

                    if(upfpsp(ia)%lllso(j).eq.0) then
                        nref0_type(ia) = nref0_type(ia) + upfpsp(ia)%lll(j)*2 +1
                    endif

                    if(is_SO.eq.0.and.upfpsp(ia)%lllso(j).eq.0) then
                        num_tmp=num_tmp+1
                    endif
                end do

                if(is_SO.eq.0) upfpsp(ia)%nbeta=num_tmp

            else ! else if (ido_DOS_999 .eq. 0)
                if(upfpsp(ia)%nwfc.gt.0) then
                    do j = 1, upfpsp(ia)%nwfc
                        nref_type(ia) = nref_type(ia) + upfpsp(ia)%lchi(j)*2 +1
                        nref0_type(ia) = nref0_type(ia) + upfpsp(ia)%lchi(j)*2 +1
                    end do
                else
                    nref_type(ia)=16   ! we will construct s, p, d, f potential
                    nref0_type(ia)=16   ! we will construct s, p, d, f potential
                endif
            endif

            z_type(ia) = upfpsp(ia)%zp
            z_type_987(ia)=z_type(ia)
            Ecut_type(ia) = upfpsp(ia)%ecutwfc
            amass_type(ia) = upfpsp(ia)%mass
            !lll_type(:,ia) = upfpsp(ia)%lll
            !nbeta_type(:,ia) = upfpsp(ia)%nbeta

        enddo
#ifdef TIMING

        if ( inode_tot == 1) then
            do ia =  1, ntype
                write ( 6, * ) "iiatom, ia", ia, iiatom(ia)
                write ( 6, * ) "nref_type, ia", ia, nref_type(ia),  nref0_type(ia)
                write ( 6, * ) "ztype, ia", ia, z_type(ia)
                write ( 6, * ) "Ecut_type, ia", ia, Ecut_type(ia)
                write ( 6, * ) "amass_type, ia", ia, amass_type(ia)
            enddo
        endif
#endif
        nref_tot_8=0
        nref_tot_beta_8=0
        mref_8=0
        mref_beta_8=0
        TOTNEL=0.d0
        do ia=1,natom
            iref_start_8(ia)=nref_tot_8
            iitype=0
            do itype=1, ntype
                if(iatom(ia).eq.iiatom(itype)) iitype=itype
            enddo
            if(iitype.eq.0) then
                write (tmp_char, *) ia
                message = 'The '//trim(adjustl(tmp_char))//' Atom "'//trim(elements(iatom(ia))%label)//'" in "'//trim(f_xatom)//'" Not Found In The Pseudopotential File. '//'Please Check The "etot.input-->IN.PSP" Setting '//'Or Check The Corresponding Pseudopotential File For The Tag "element"'
                !write(message,*) "itype not found, stop", iatom(ia),ia
                call error_stop(message,__FILE__,__LINE__)
            endif
            ityatom(ia)=iitype
            numref(ia)=nref_type(iitype)
            numref0(ia)=nref0_type(iitype)

            iMDatom(ia)=amass_type(iitype)
            mass_type(iitype) = amass_type(iitype)
            TOTNEL = TOTNEL + z_type(iitype)
            nref_tot_8=nref_tot_8+nref_type(iitype)
            if(nref_type(iitype).gt.mref_8) mref_8=nref_type(iitype)
        enddo  ! enddo do ia=1.natom
        totNel_987_neutral=TOTNEL
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !          !Weile Jia, add Qij_pd_1
        !          IF (ipsp_all .eq. 2) THEN
        !              CALL read_key_words( 9, "QIJ_PD",LEN('QIJ_PD'),right,IERR)
        !              IF(IERR .eq. 0) then
        !                  read (right, *)  QIJ_PD_1 
        !              ELSE
        !                  QIJ_PD_1 = 0
        !              ENDIF
        !
        !              if(inode_tot.eq.1) then
        !                  write(22,1923) QIJ_PD_1
        !              endif
        !1923      format(" QIJ_PD    = ",i2)
        !          ENDIF
        !!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !!ccc This is for ultra-soft Qij(r) implementation
        !!ccc either G-space (iQijL0_GS=1), or real space (wmask_q) (iQijL0_GS=2)
        !          CALL read_key_words ( 9, 'QijL0_GS', LEN('QijL0_GS'),right, IERR )
        !          if(ierr.eq.0) then
        !              READ ( right, * ) iQijL0_GS
        !          else
        !              iQijL0_GS = 1     ! Gspace implementation, Real-space: 2
        !          endif
        !
        !          if(inode_tot.eq.1) then
        !              write(22,1116) iQijL0_GS
        !          endif
        !1116   format(" QijL0_GS  = ",i2)
        CALL read_key_words (9, 'RANDOM_SEED', LEN('RANDOM_SEED'), right, ierr)
        if (ierr == 0) then
            read(right,*) irandom_seed
            write(22,*) "RANDOM_SEE=", irandom_seed
        else
           irandom_seed=0
        endif


        CALL read_key_words (9, 'QIJ_DETAIL', LEN('QIJ_DETAIL'), right, ierr)
        if (ierr == 0) then
            read (right, *) QIJ_PD_1, iQijL0_GS
        else
            QIJ_PD_1 = 0
            iQijL0_GS = 1
        end if
        if (inode_tot == 1) then
            write (22, '(" QIJ_DETAIL = ", 2i3)') QIJ_PD_1, iQijL0_GS
        end if
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, "SCF_MIX", LEN('SCF_MIX'), right, IERR )
        if(ierr.eq.0) then
            read ( right, * ) temp_char
            mixing = temp_char
            call transform_to_upper(mixing)
            if (index(mixing,"CHA")>0)then
                potential_mix = .false.
            elseif ( index(mixing,"POT")>0) then
                potential_mix = .true.
                !else
                !write ( 6, * ) "MIX should be CHARGE/POTENTIAL "
                !if(inode_tot .eq. 1) then
                !    call mpi_abort(mpi_comm_world,ierr)
                !endif

            end if
            if(potential_mix.eq. .true.)then
                potential_mix = .false.
                write ( 6, * ) "POTENTIAL MIX no longer supported"
            endif
        else
            potential_mix = .false.
            mixing = "CHARGE"
        endif

        !      if(inode_tot.eq.1) then
        !          write(22,*) "SCF_MIX       = ", trim(mixing)
        !      endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, "PWSCF_OUTPUT", LEN('PWSCF_OUTPUT'),right, ierror )
        IF ( ierror == 0 ) THEN
            READ ( right, * ) PWSCF_OUTPUT
        ELSE
            PWSCF_OUTPUT = .FALSE.
        ENDIF
        IF (inode_tot == 1) THEN
            WRITE (temp_char, *) PWSCF_OUTPUT
            WRITE (22, *) 'PWSCF_OUTPUT     =  '//TRIM(ADJUSTL(temp_char)) 
        ENDIF
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, "Ecut", LEN('Ecut'), right, IERR )
        if(ierr.eq.0) then
            read ( right, * ) Ecut
        else
            Ecut=-1.d0
            do itype=1,ntype
                if(Ecut_type(itype).gt.Ecut) Ecut=Ecut_type(itype)
            enddo
            !if(ACCURACY .eq. "VERYHIGH") then  !change Ecut if very high accuracy
            !    if(ipsp_all.eq.1) then
            !        Ecut = Ecut * 1.20
            !    else
            !        Ecut = Ecut * 1.10
            !    endif
            !endif
        endif

        if(inode_tot.eq.1) then
            write(22,*) "Ecut      = ", Ecut
        endif

        CALL read_key_words ( 9, "Ecut2", LEN('Ecut2'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) Ecut2
        else
            if (ipsp_all.eq.1) then
                Ecut2 = 2*Ecut                             ! we might need to use Ecut2 =2Ecut
            else
                Ecut2 = Ecut*maxval(upfpsp(1:ntype)%ecut2/upfpsp(1:ntype)%ecutwfc)
            end if
            if(ACCURACY .eq. "HIGH" .or. ACCURACY .eq. "VERYHIGH") Ecut2 = 4 * Ecut  !change Ecut if high accuracy
            !if(trim(JOB).eq."RELAX" .or. trim(JOB) .eq. "NEB") Ecut2 = 4 * Ecut
        endif


        if(inode_tot.eq.1) then
            write(22,*) "Ecut2     = ", Ecut2
        endif

        CALL read_key_words ( 9, "Ecut2L", LEN('Ecut2L'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) Ecut2L
        else
            if (ipsp_all.eq.1) then
                Ecut2L=Ecut2
            else
                Ecut2L=4*Ecut2
            endif
            !if(ACCURACY .eq. "HIGH") Ecut2L = 2 * Ecut2 
            if(ACCURACY .eq. "VERYHIGH") Ecut2L = 4 * Ecut2 
            if(PWSCF_OUTPUT) then
                Ecut2L=Ecut2
            endif
        endif
        if(PWSCF_OUTPUT) then
            Ecut2L=Ecut2
        endif

        if(inode_tot.eq.1) then
            write(22,*) "Ecut2L    = ", Ecut2L
        endif


        CALL read_key_words ( 9, "EcutP", LEN('EcutP'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) EcutP
        else
            EcutP=Ecut
            if(ACCURACY .eq. "HIGH") EcutP = 4 * Ecut 
            if(ACCURACY .eq. "VERYHIGH") EcutP = 4 * Ecut 
        endif

        if(inode_tot.eq.1) then
            write(22,*) "EcutP    = ", EcutP
        endif

        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !**** change Ecut to A.U
        Ecut=Ecut/2
        Ecut2=Ecut2/2
        Ecut2L=Ecut2L/2
        EcutP=EcutP/2

        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, "N123", LEN('N123'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) n1, n2, n3
        else
            fackpt=dsqrt(2.d0*Ecut2)/(4*datan(1.d0))
            dd11=fackpt*dsqrt(AL(1,1)**2+AL(2,1)**2+AL(3,1)**2)
            dd22=fackpt*dsqrt(AL(1,2)**2+AL(2,2)**2+AL(3,2)**2)
            dd33=fackpt*dsqrt(AL(1,3)**2+AL(2,3)**2+AL(3,3)**2)

            nnnstart = 2
            call get_n123(n2,dd22,nnnstart)
            nnnstart = 2
            call get_n123(n3,dd33,nnnstart)
            nnnstart = 2
            call get_n123(n1,dd11,nnnstart) 
            n1_init = n1
            if (mod(n1_init,nnodes_tot) /= 0) then
                do
                    nnnstart = nnnstart + 1
                    if (nnnstart > 614) exit
                    if (mod(n1,nnodes_tot) == 0) then
                        exit
                    else
                        call get_n123(n1,dd11,nnnstart)
                    end if
                end do
                if (n1-n1_init > 12) n1 = n1_init + nnodes_tot - mod(n1_init,nnodes_tot)
                if (n1-n1_init == 0) n1 = n1_init + nnodes_tot - mod(n1_init,nnodes_tot)
            end if
            if (n1_init == n2) n2 = n1
            if (n1_init == n3) n3 = n1

            if(PWSCF_OUTPUT) then
                call init_pwscf_unit(AL)
                call pwscf_grid_init(n1,n2,n3, at, bg, gcutm)
            endif
        endif
        if(PWSCF_OUTPUT) then
            n1=good_fft_order(n1)
            n2=good_fft_order(n2)
            n3=good_fft_order(n3)
        endif

        !    n1=n1+nnodes_b-mod(n1,nnodes_b), this is the wrong way to make the
        !    n1 divisable by nnodes

        if(inode_tot.eq.1) then
            write(22,1111) n1,n2,n3
        endif
        1111   format(" N123      = ",i5,1x,i5,1x,i5)
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        ALX = sqrt( AL(1,1)**2 + AL(2,1)**2 + AL(3,1)**2) * A_AU_1
        ALY = sqrt( AL(1,2)**2 + AL(2,2)**2 + AL(3,2)**2) * A_AU_1
        ALZ = sqrt( AL(1,3)**2 + AL(2,3)**2 + AL(3,3)**2) * A_AU_1

        standard_al = 6.0
        factor = 0.80
        N1S = N1
        N2S = N2
        N3S = N3

        !if((2*standard_al).le.(factor*ALX)) N1S = (standard_al*N1) / ALX
        !if((2*standard_al).le.(factor*ALY)) N2S = (standard_al*N2) / ALY
        !if((2*standard_al).le.(factor*ALZ)) N3S = (standard_al*N3) / ALZ
        !write(*,*) "standard_al=",standard_al,ALX,ALY,ALZ
        !N1S = (standard_al*N1) / ALX
        !N2S = (standard_al*N2) / ALY
        !N3S = (standard_al*N3) / ALZ
        
        h1=det(AL)/cross(AL(:,2),AL(:,3))*A_AU_1
        h2=det(AL)/cross(AL(:,1),AL(:,3))*A_AU_1
        h3=det(AL)/cross(AL(:,1),AL(:,2))*A_AU_1

        N1S=(standard_al/h1*N1)
        N2S=(standard_al/h2*N2)
        N3S=(standard_al/h3*N3)

        call get_n456(N1S)
        call get_N456(N2S)
        call get_N456(N3S)

        CALL read_key_words( 9, 'NS123', LEN('NS123'), right, IERR)
        if(ierr .eq. 0) then
            READ ( right, *)  N1S, N2S, N3S
        endif

        if(inode_tot.eq.1) then
            write(22,1112) N1S, N2S, N3S
        endif
        1112   format(" NS123     = ",i5,1x,i5,1x,i5)

        ALs(:,1)=(AL(:,1)*n1s)/n1
        ALs(:,2)=(AL(:,2)*n2s)/n2
        ALs(:,3)=(AL(:,3)*n3s)/n3

        !write(*,*) "oh1=",det(AL)/cross(AL(:,2),AL(:,3))*A_AU_1
        !write(*,*) "oh2=",det(AL)/cross(AL(:,1),AL(:,3))*A_AU_1
        !write(*,*) "oh3=",det(AL)/cross(AL(:,1),AL(:,2))*A_AU_1
        !write(*,*) "h1=",det(ALs)/cross(ALs(:,2),ALs(:,3))*A_AU_1
        !write(*,*) "h2=",det(ALs)/cross(ALs(:,1),ALs(:,3))*A_AU_1
        !write(*,*) "h3=",det(ALs)/cross(ALs(:,1),ALs(:,2))*A_AU_1

        MCTRL_AL=AL
        AL0=AL
        ALs0=ALs
        n1s_z0=n1s
        n2s_z0=n2s
        n3s_z0=n3s
        n1_z0=n1
        n2_z0=n2
        n3_z0=n3
        STRESS_EXT_AL0=AL0
        call get_ALI(AL0,ALI0)
        call get_ALI(ALs0,ALIs0)
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, "N123L", LEN('N123L'), right, IERR )
        if(ierr.eq.0) then  ! there is a input from etot.input
            READ ( right, * ) n1L, n2L, n3L
        else       ! there is no input from etot.input, use default
            !N123L_scale_factor=dsqrt(Ecut2L/Ecut2)
            !n1L=N123L_scale_factor*n1
            !n2L=N123L_scale_factor*n2
            !n3L=N123L_scale_factor*n3
            if(abs(Ecut2L-Ecut2)<1.d-5) then
                n1L=n1
                n2L=n2
                n3L=n3
            else if(abs(Ecut2L-4*Ecut2)<1.d-5) then
                n1L=2*n1
                n2L=2*n2
                n3L=2*n3
            else
                fackpt=dsqrt(2.d0*Ecut2L)/(4*datan(1.d0))
                dd11=fackpt*dsqrt(AL(1,1)**2+AL(2,1)**2+AL(3,1)**2)
                dd22=fackpt*dsqrt(AL(1,2)**2+AL(2,2)**2+AL(3,2)**2)
                dd33=fackpt*dsqrt(AL(1,3)**2+AL(2,3)**2+AL(3,3)**2)

                nnnstart = 2
                call get_n123(n2L,dd22,nnnstart)
                nnnstart = 2
                call get_n123(n3L,dd33,nnnstart)
                nnnstart = 2
                call get_n123(n1L,dd11,nnnstart) 
                n1_init = n1L
                if (mod(n1_init,nnodes_tot) /= 0) then
                    do
                        nnnstart = nnnstart + 1
                        if (nnnstart > 614) exit
                        if (mod(n1L,nnodes_tot) == 0) then
                            exit
                        else
                            call get_n123(n1L,dd11,nnnstart)
                        end if
                    end do
                    if (n1L-n1_init > 12) n1L = n1_init + nnodes_tot - mod(n1_init,nnodes_tot)
                    if (n1L-n1_init == 0) n1L = n1_init + nnodes_tot - mod(n1_init,nnodes_tot)
                end if
                if (n1_init == n2L) n2L = n1L
                if (n1_init == n3L) n3L = n1L
                !
                if(n1L<n1) n1L=n1
                if(n2L<n2) n2L=n2
                if(n3L<n3) n3L=n3
            endif
            !
        endif
        if(PWSCF_OUTPUT) then
            n1L=n1
            n2L=n2
            n3L=n3
        endif

        if(inode_tot.eq.1) then
            write(22,1113) n1L,n2L,n3L
        endif
        1113   format(" N123L     = ",i5,1x,i5,1x,i5)
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        nkpt_extend=1
        CALL read_key_words ( 9, "MP_N123", LEN('MP_N123'), right, IERR)
        if(ierr.eq.0) then
            gen_from_mp = 1
            read(right,*,iostat=ierror) nk1,nk2,nk3,sk1,sk2,sk3,iflag_symm
            MCTRL_mpn123_nk1=nk1
            MCTRL_mpn123_nk2=nk2
            MCTRL_mpn123_nk3=nk3
            MCTRL_mpn123_sk1=sk1
            MCTRL_mpn123_sk2=sk2
            MCTRL_mpn123_sk3=sk3
            ! here, nk1,2,3,sk1,2,3, are not really used, it will be readin
            ! from gen_kpt_symm again
            ! here we only want iflag_symm
            nkpt_extend=nk1*nk2*nk3
            if(ierror.ne.0) iflag_symm=0
            if(iflag_symm.ne.2.and.is_SOM.eq.1) then
                if(inode_tot .eq. 1) then
                    write(6,*) "**********************************************************"
                    write(6,*) "WARNING: SPIN=222, SHOULD NOT USE SYMM in MP_N123, COULD BE WRONG"
                    write(6,*) "**********************************************************"
                    write(22,*) "**********************************************************"
                    write(22,*) "WARNING: SPIN=222, SHOULD NOT USE SYMM in MP_N123, COULD BE WRONG"
                    write(22,*) "**********************************************************"
                endif
            endif


            if(inode_tot .eq. 1) then
                write (6, *) 'The kpoints were generated from MP_N123'
                ! zhilin, add the generation of IN.KPT and IN.SYMM
            endif
            if(inode_tot.eq.1) then
                write(22,*) "MP_N123 =", trim(right)
            endif
        else
            gen_from_mp = 0
            MCTRL_mpn123_nk1=1
            MCTRL_mpn123_nk2=1
            MCTRL_mpn123_nk3=1
            MCTRL_mpn123_sk1=0
            MCTRL_mpn123_sk2=0
            MCTRL_mpn123_sk3=0
        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, "STRESS_CORR", LEN('STRESS_CORR'), right, IERR)

        if(ierr.eq.0) then
            read (right,*)stress_corr_n1,stress_corr_E1,stress_corr_n2,stress_corr_E2
        else
            stress_corr_n1=1
            stress_corr_n2=2
            stress_corr_E1=0.d0
            stress_corr_E2=0.d0
        endif
        if(inode_tot.eq.1) then
            write(22, "(a,4E14.6)") " STRESS_CORR = ", stress_corr_n1,stress_corr_E1,stress_corr_n2,stress_corr_E2
        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, "IN.EXT_FORCE", LEN('IN.EXT_FORCE'), right, IERR)
        iflag_ext_force=0
        if(ierr.eq.0) then
            read(right,*) right_logical
            if(right_logical) then
                iflag_ext_force=1
                allocate(force_ext(3,natom))
                open(10,file="IN.EXT_FORCE")
                rewind(10)
                read(10,*) natom_tmp
                if(natom_tmp.ne.natom) then
                    write(6,*) "natom in IN.EXT_FORCE NOT CORRECT", natom_tmp
                    stop
                endif
                do iat=1,natom
                    read(10,*)iat_tmp,force_ext(1,iat),force_ext(2,iat),force_ext(3,iat)
                enddo
                force_ext=force_ext*0.529177d0/27.211396d0
                close(10)
            endif
            if(inode_tot.eq.1) then
                write(22,*) "IN.EXT_FORCE =", trim(right)
            endif
        endif

        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc




        CALL read_key_words ( 9, "EGG_CORR", LEN('EGG_CORR'), right, IERR)

        if(ierr.eq.0) then
            read ( right, * ) right_logical
            if(right_logical) then
                if(trim(JOB).eq."EGGFIT") then
                    write(6,*) "JOB=EGGFIT, should not have EGG_CORR=T"
                    call MPI_ABORT(MPI_COMM_WORLD,ierr)
                endif
                iflag_egghead_calc=1
                open(10,file="CC.egghead")
                rewind(10)
                read(10,*) ntype_tmp,np1_egg,np2_egg,np3_egg
                if(ntype.ne.ntype_tmp) then
                    write(6,*) "ntype.ne.ntype_tmp from CC.egghead,stop"
                    write(6,*) ntype,ntype_tmp
                    stop
                endif

                read(10,*) AL_grid(1,1),AL_grid(2,1),AL_grid(3,1)
                read(10,*) AL_grid(1,2),AL_grid(2,2),AL_grid(3,2)
                read(10,*) AL_grid(1,3),AL_grid(2,3),AL_grid(3,3)
                AL_grid(:,1)=AL_grid(:,1)*n1L/A_AU_1
                AL_grid(:,2)=AL_grid(:,2)*n2L/A_AU_1
                AL_grid(:,3)=AL_grid(:,3)*n3L/A_AU_1
                sum=0.d0
                do i1=1,3
                    do i2=1,3
                        sum=sum+abs(AL_grid(i1,i2)-AL(i1,i2))
                    enddo
                enddo
                if(sum.gt.1.E-4) then
                    write(6,*) "AL_grid in CC.egghead not right, stop"
                    stop
                endif
                read(10,*) Ecut_tmp,Ecut2_tmp,Ecut2L_tmp
                Ecut_tmp=Ecut_tmp/2
                Ecut2_tmp=Ecut2_tmp/2
                Ecut2L_tmp=Ecut2L_tmp/2
                if(abs(Ecut_tmp-Ecut)+abs(Ecut2_tmp-Ecut2)+abs(Ecut2L_tmp-Ecut2L).gt.1.D-2) then
                    write(6,*) "Ecut,Ecut2,Ecut2L in CC.egghead not right,stop"
                    stop
                endif


                do I = 1,ntype
                    read(10,*) filename
                    if(trim(vwr_atom(i)).ne.trim(filename)) then
                        write(6,*) "psp filename diff in CC.egghead,stop"
                        write(6,*) vwr_atom(i)
                        write(6,*) filename
                        stop
                    endif
                enddo

                mp1=np1_egg/2
                mp2=np2_egg/2
                mp3=np3_egg/2
                allocate(Egg_corr_coeff(0:mp1,0:mp2,0:mp3,ntype_tmp,3))
                do ii=1,ntype
                    do jj=1,(1+mp1)*(1+mp2)*(1+mp3)
                        read(10,*) itype,i1,i2,i3,E_tmp1,E_tmp2,E_tmp3
                        Egg_corr_coeff(i1,i2,i3,itype,1)=E_tmp1
                        Egg_corr_coeff(i1,i2,i3,itype,2)=E_tmp2
                        Egg_corr_coeff(i1,i2,i3,itype,3)=E_tmp3
                    enddo
                enddo
                close(10)

            else
                iflag_egghead_calc=0
            endif
            if(inode_tot.eq.1) then
                write(22,*) "EGG_CORR =", trim(right)
            endif
        else
            iflag_egghead_calc=0
        endif

        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        !
        igga=0
        !hse control flags
        xctype = -1
        iflag_hse = 0
        use_hse = 0
        use_hse_thisEtot=0
        use_sbfft_hse = 1
        iflag_hse_dist = 1
        iflag_rVV10=0
        iflag_LDAWKM=0
        CALL read_key_words ( 9,"XCFUNCTIONAL",LEN('XCFUNCTIONAL'),right,IERR )
                       if(trim(JOB).eq."MOMENT") then   
                        ! This is crazy. there is a bug in scfft_moment inside scfft_cu, setting
                        ! it as HSE will make it work, this is a temp solution. 
                        ! Some initial setup is not correct
                        right = "HSE"
                        ierr=0
                        endif
        if ( ierr == 0 ) then
            read ( right, '(A200)' ) temp_char_xc
            temp_char_xc = ADJUSTL(temp_char_xc)
            call capital(temp_char_xc, len(temp_char_xc))
            tmp_char = temp_char_xc
            if ( index(temp_char_xc,"LDAWKM2") == 1 ) then  ! this is for SCF WKM calculation
              ! Note, even if it uses LDAWKM2, it will still contain LDAKWM,
              ! index(LDAWKM2,KDAWKM)=1, so must test test the long version !
              ! It is actually reallyt dangerous to use this way. 
                  if(iflag_wkm.eq.1) then
                  write(22,*) "for XCFUNCTIONAL=LDAWKM, must use JOB=SCF,stop" 
                  stop
                  endif

                xctype(1) = 100
                xctype(2) = 1
                xctype(3) = 9
                get_dft_short = "slater-dirac-exchange" ! for DFT3 use
                iflag_LDAWKM=1
                itype_LDAWKM=2
             elseif ( index(temp_char_xc,"LDAWKM") == 1 ) then  ! this is for SCF WKM calculation
                  if(iflag_wkm.eq.1) then
                  write(22,*) "for XCFUNCTIONAL=LDAWKM2, must use JOB=SCF,stop" 
                  stop
                  endif
                xctype(1) = 100
                xctype(2) = 1
                xctype(3) = 9
                get_dft_short = "slater-dirac-exchange" ! for DFT3 use
                iflag_LDAWKM=1
                itype_LDAWKM=1


             elseif ( index(temp_char_xc,"PBEWKM2") == 1 ) then  ! this is for SCF WKM calculation
                  if(iflag_wkm.eq.1) then
                  write(22,*) "for XCFUNCTIONAL=PBEWKM, must use JOB=SCF,stop" 
                  stop
                  endif
                xctype(1) = 100
                xctype(2) = 101
                xctype(3) = 130
                get_dft_short = "slater-dirac-exchange" ! for DFT3 use
                iflag_LDAWKM=1
                itype_LDAWKM=2
             elseif ( index(temp_char_xc,"PBEWKM") == 1 ) then  ! this is for SCF WKM calculation
                  if(iflag_wkm.eq.1) then
                  write(22,*) "for XCFUNCTIONAL=PBEWKM2, must use JOB=SCF,stop" 
                  stop
                  endif
                xctype(1) = 100
                xctype(2) = 101
                xctype(3) = 130
                get_dft_short = "slater-dirac-exchange" ! for DFT3 use
                iflag_LDAWKM=1
                itype_LDAWKM=1

            elseif ( index(temp_char_xc,"LDA") == 1 ) then
                xctype(1) = 100
                xctype(2) = 1
                xctype(3) = 9
                get_dft_short = "slater-dirac-exchange" ! for DFT3 use
            elseif ( index(temp_char_xc,"SX") == 1 ) then
                hyb_gga_type=0
                use_hse = 1
                use_hse_thisEtot = 1
            elseif ( index(temp_char_xc,"HSE") == 1 ) then
                hyb_gga_type=1
                use_hse = 1
                use_hse_thisEtot = 1
                xctype(1) = 100
                xctype(2) = 428
                xctype(3) = 428
                get_dft_short = "hse06" ! for DFT3 use
            elseif ( index(temp_char_xc,"BP") == 1 ) then
                xctype(1) = 100
                xctype(2) = 106
                xctype(3) = 132
                get_dft_short = "bp" ! for DFT3 use
            elseif ( index(temp_char_xc,"PW91") == 1 ) then
                xctype(1) = 100
                xctype(2) = 109
                xctype(3) = 134
                get_dft_short = "pw91" ! for DFT3 use
            elseif ( index(temp_char_xc,"PBESOL") == 1 ) then
                xctype(1) = 100
                xctype(2) = 116
                xctype(3) = 133
                get_dft_short = "pbesol" ! for DFT3 use
            elseif ( index(temp_char_xc,"PBE") == 1 ) then
                xctype(1) = 100
                xctype(2) = 101
                xctype(3) = 130
                igga=1
                get_dft_short = "pbe" ! for DFT3 use
            elseif ( index(temp_char_xc,"GGA") == 1 ) then
                xctype(1) = 100
                xctype(2) = 101
                xctype(3) = 130
                igga=1
                get_dft_short = "pbe" ! for DFT3 use
            elseif ( index(temp_char_xc,"WC") == 1 ) then
                xctype(1) = 100
                xctype(2) = 118
                xctype(3) = 130
                get_dft_short = "wc" ! for DFT3 use
            elseif ( index(temp_char_xc,"TPSS") == 1 ) then
                xctype(1) = 100
                xctype(2) = 202
                xctype(3) = 231
                get_dft_short = "tpss" ! for DFT3 use
            elseif ( index(temp_char_xc,"SCAN") == 1 ) then
                xctype(1) = 100
                xctype(2) = 263
                xctype(3) = 267
                !   elseif ( index(temp_char_xc,"RVV10") == 1 ) then
                !       xctype(1) = 100
                !       xctype(2) = 1
                !       xctype(3) = 9
                !       iflag_rVV10=1
            elseif ( index(temp_char_xc,"LIBXC") > 0 ) then
                xctype(1) = 100
                i_plus = index(temp_char_xc,"+")
                if ( i_plus < 1 ) then
                    read ( temp_char_xc(7:i_plus-1), * ) xctype(2)
                    xctype(3) = xctype(2)
                else
                    read ( temp_char_xc(7:i_plus-1), * ) xctype(2)
                    temp_char_xc = temp_char_xc(i_plus+1:)
                    temp_char_xc = adjustl(temp_char_xc)
                    read ( temp_char_xc(7:), * ) xctype(3)
                end if
            elseif ( index(temp_char_xc,"XC") > 0 ) then
                xctype(1) = 100
                i_plus = index(temp_char_xc,"+")
                !if ( i_plus < 1 ) then
                !    xctype(2) = libxc_id(temp_char_xc)
                !    xctype(3) = 0
                !else
                !    xctype(2) = libxc_id(temp_char_xc(:i_plus-1))
                !    temp_char_xc = temp_char_xc(i_plus+1:)
                !    temp_char_xc = adjustl(temp_char_xc)
                !    xctype(3) = libxc_id(trim(temp_char_xc))
                !end if
                if ( i_plus < 1 ) then
                    xctype(2) = xc_f90_functional_get_number(temp_char_xc(4:))
                    xctype(3) = xctype(2)
                else
                    xctype(2) = xc_f90_functional_get_number(temp_char_xc(4:i_plus-1))
                    temp_char_xc = temp_char_xc(i_plus+1:)
                    temp_char_xc = adjustl(temp_char_xc)
                    xctype(3) = xc_f90_functional_get_number(temp_char_xc(4:))
                end if
            else
                write ( 6, * ) 'XCFUNCTIONAL should be LDA/GGA/...'
                write ( 6, * ) 'For libxc: "XC_XXX" or "LIBXC_YYY" '
                if(inode_tot .eq. 1) then
                    call mpi_abort(mpi_comm_world,ierr)
                endif

            end if

            if ( index(temp_char_xc,"RVV10").gt.0 ) then
                iflag_rVV10=1
            endif

        else
            tmp_char = "PBE"
            xctype(1) = 100
            xctype(2) = 101
            xctype(3) = 130
            igga=1
            xgga=0.d0
            get_dft_short = "pbe" ! for DFT3 use
        end if




        if(inode_tot.eq.1) then
            write (22, *) "XCFUNCTIONAL = ", trim(tmp_char)
        endif
        !-------------------------------------------------------------
        if(iflag_rVV10.eq.1) then
            CALL read_key_words ( 9, 'RVV10_DETAIL', LEN('RVV10_DETAIL'),right, IERR )
            if(ierr.eq.0) then
                READ (right,*) b_rVV10,c_rVV10
            else
                b_rVV10=6.3d0
                c_rVV10=0.0093
            endif
            if(inode_tot.eq.1) then
                write(22,*) "RVV10_DETAIL  = ", b_rVV10,c_rVV10
            endif
        endif

        !-------------------------------------------------------------

        if( use_hse .eq. 1) then
            CALL read_key_words ( 9, 'HSE_SAVE_CPU_MEMORY', LEN('HSE_SAVE_CPU_MEMORY'),right, IERR )
            IF(ierr.eq.0) then
                read ( right, * ) right_logical
                IF ( right_logical ) THEN
                    iflag_hse_dist = 1
                ELSE
                    iflag_hse_dist = 0
                ENDIF
            ELSE
                iflag_hse_dist = 1
                right_logical = .TRUE.
            ENDIF
            if(inode_tot.eq.1) then
                write(22,*) "HSE_SAVE_CPU_MEMORY = ", right_logical
            endif
        endif

        !igga=0
        xgga=0.d0
        !!! if JOB = NONSCF and XCFUNCTIONAL = HSE, this subroutine will check the whether the OUT.HSEWR1 exists!!!
        if (JOB_1 == "NONSCF" .and. use_hse == 1) call check_hsewr()
        !if (use_hse == 1 .and. is_SOM .eq. 1) then
        !    message="HSE+SOC only support SPIN=22"
        !    call error_stop(message,__FILE__,__LINE__)
        !endif 

        if(interp_DOS_999.eq.1) then  ! especial arrangement, use HSE facility for interp_DOS_999
            hyb_gga_type=1
            use_hse = 1
            use_hse_thisEtot = 1
            xctype(1) = 100
            xctype(2) = 428
            xctype(3) = 428
        endif

        if ( xctype(1) == 100 ) call libxc_check(xctype(2),xctype(3),islda)
        !
        !
        if(igga.ne.0.and.igga.ne.1.and.inode.eq.1) then
            !write(6,*) "XcFunctional must be 0(no gga) or 1(gga), stop",igga
            !call mpi_abort(MPI_COMM_WORLD,ierr)
        endif
        ! set hse options 
        if(use_hse .eq. 1) then
            if(ipsp_all .eq. 2) then
                !if(precision_flag.ne.single_prec .and. precision_flag .ne. auto_prec) then
                !    message="HSE for ultra-soft pseudopotential can only run with PRECISION=SINGLE or AUTO"
                !    call error_stop(message,__FILE__,__LINE__)
                !endif
                message="HSE not support ultra-soft pseudopotential"
                call error_stop(message,__FILE__,__LINE__)
            endif
            CALL read_key_words ( 9, "HSE_OMEGA", LEN('HSE_OMEGA'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, * ) hse_omega_bohr
                if(hse_omega_bohr.lt.1.D-5) hse_omega_bohr=1.D-5
            else
              hse_omega_bohr = 0.20  ! This is bohr, eq=1.058 1/A for HSE06
            endif

            if(inode_tot.eq.1) then
                write(22,*) "HSE_OMEGA  = ", hse_omega_bohr
            endif
            HSE_OMEGA = HSE_omega_bohr  * A_AU_1

            CALL read_key_words ( 9, "HSE_ALPHA", LEN('HSE_ALPHA'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, * ) hse_alpha
            else
                hse_alpha = 0.25
            endif

            if(inode_tot.eq.1) then
                write(22,*) "HSE_ALPHA  = ", hse_alpha
            endif

            CALL read_key_words ( 9, "HSE_BETA", LEN('HSE_BETA'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, * ) hse_beta
            else
                hse_beta = 0.0
            endif

            if(inode_tot.eq.1) then
                write(22,*) "HSE_BETA  = ", hse_BETA
            endif



            x_gamma_extrapolation=1
            CALL read_key_words ( 9, "NQ123", LEN('NQ123'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, * ) hse_nq1,hse_nq2, hse_nq3
            else
                CALL read_key_words ( 9, "MP_N123", LEN('MP_N123'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * ) hse_nq1,hse_nq2, hse_nq3
                else
                    hse_nq1=1
                    hse_nq2=1
                    hse_nq3=1  
                endif
            endif

            if(interp_DOS_999.eq.1) then   ! special
                hse_nq1=hse_nq1_t
                hse_nq2=hse_nq2_t
                hse_nq3=hse_nq3_t
            endif

            !if(inode_tot.eq.1) write(22,1142) hse_nq1,hse_nq2,hse_nq3
            1142 format(" NQ123      = ",i5,1x,i5,1x,i5)

            if(use_sbfft_hse .eq. 1) then  !! weile, calculate the small box FFT size.
                fackpt=dsqrt(2.d0*EcutP)/(4*datan(1.d0))
                dd11=fackpt*dsqrt(AL(1,1)**2+AL(2,1)**2+AL(3,1)**2) + 2
                dd22=fackpt*dsqrt(AL(1,2)**2+AL(2,2)**2+AL(3,2)**2) + 2
                dd33=fackpt*dsqrt(AL(1,3)**2+AL(2,3)**2+AL(3,3)**2) + 2
                nnnstart = 2
                call get_n123(hse_n1,dd11,nnnstart) 
                nnnstart = 2
                call get_n123(hse_n2,dd22,nnnstart) 
                nnnstart = 2
                call get_n123(hse_n3,dd33,nnnstart) 
                !
                n1_init = hse_n1
                if (mod(n1_init,nnodes_tot) /= 0) then
                    do
                        nnnstart = nnnstart + 1
                        if (nnnstart > 614) exit
                        if (mod(hse_n1,nnodes_tot) == 0) then
                            exit
                        else
                            call get_n123(hse_n1,dd11,nnnstart)
                        end if
                    end do
                    if (hse_n1-n1_init > 12) hse_n1 = n1_init + nnodes_tot - mod(n1_init,nnodes_tot)
                    if (hse_n1-n1_init == 0) hse_n1 = n1_init + nnodes_tot - mod(n1_init,nnodes_tot)
                end if
                if (n1_init == hse_n2) hse_n2 = hse_n1
                if (n1_init == hse_n3) hse_n3 = hse_n1
                !
                if(hse_n1 .gt. n1)  hse_n1 = n1
                if(hse_n2 .gt. n2)  hse_n2 = n2
                if(hse_n3 .gt. n3)  hse_n3 = n3
                !hse_n1 = (n1+1)*2.d0/3 + 1
                !hse_n2 = (n2+1)*2.d0/3 + 1
                !hse_n3 = (n3+1)*2.d0/3 + 1
                CALL read_key_words ( 9, "P123", LEN('P123'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * ) hse_n1,hse_n2, hse_n3
                endif

                if(ipsp_all .eq. 2) then
                    hse_n1 = n1
                    hse_n2 = n2
                    hse_n3 = n3
                endif
             !   if(ACCURACY .eq. "HIGH") then
             !       hse_n1 = n1
             !       hse_n2 = n2
             !       hse_n3 = n3
             !   endif
                if(inode_tot.eq.1) write(22,1141) hse_n1,hse_n2,hse_n3
                1141            format(" P123      = ",i5,1x,i5,1x,i5)
            else
                hse_n1 = n1
                hse_n2 = n2
                hse_n3 = n3
            endif

        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !          CALL read_key_words ( 9, "HSE_MIX", LEN('HSE_MIX'), right, IERR )
        !          if(ierr.eq.0) then
        !              !READ ( right, * ) ihse_PBE_scf,hse_mix,max_sxp,tolHSE_mix
        !              READ ( right, * ) hse_mix,max_sxp,tolHSE_mix
        !              tolHSE_mix=tolHSE_mix/27.211396d0
        !          else
        !              !ihse_PBE_scf=1     ! first do one PBE Etotcalc, before HSE Etotcalc, both with niter0
        !              hse_mix=1.d0
        !              max_sxp=1
        !              tolHSE_mix=0.1E-04
        !          end if
        !          if(inode_tot.eq.1) then
        !              !write(22,1123) ihse_PBE_scf,hse_mix,max_sxp, tolHSE_mix
        !              write(22,1123) hse_mix,max_sxp, tolHSE_mix
        !              if(max_sxp.eq.0) tolHSE_mix=0.0D0
        !          endif
        !1123   format(" HSE_MIX   = ",i5, f15.8, i5, f15.8)
        !1123   format(" HSE_MIX   = ",f15.8, i5, f15.8)
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !          CALL read_key_words ( 9, "HSE_dN", LEN('HSE_dN'), right, IERR )
        !          if(ierr.eq.0) then
        !              READ ( right, * ) iupdate_sxp_dn
        !          else
        !              !iupdate_sxp_dn=3
        !              iupdate_sxp_dn=6
        !          endif
        !
        !          if(inode_tot.eq.1) then
        !              write(22,1124) iupdate_sxp_dn
        !          endif
        !1124   format(" HSE_dN    = ",i5)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        CALL read_key_words (9, 'HSE_DETAIL', LEN('HSE_DETAIL'), right, IERR)
        if (IERR == 0) then
            read (right,*,IOSTAT=ierror) hse_mix, max_sxp, tolHSE_mix, iupdate_sxp_dn, ihse_PBE_scf, icheck_dsxH
            if(ierror.ne.0) then
                read (right,*,IOSTAT=ierror) hse_mix, max_sxp, tolHSE_mix, iupdate_sxp_dn, ihse_PBE_scf
                icheck_dsxH=1
            endif
            tolHSE_mix=tolHSE_mix/27.211396d0
            if(max_sxp.eq.0) tolHSE_mix=0.0D0
        else
            hse_mix=1.d0
            max_sxp=1
            tolHSE_mix=0.d0
            iupdate_sxp_dn=6
            ihse_PBE_scf = 1
            icheck_dsxH = 1
        end if
        if(inode_tot.eq.1) then
            write(22,'(" HSE_DETAIL    = ",f15.8, i5, f15.8,i5,i2,i2)') hse_mix,max_sxp, tolHSE_mix*27.211396d0, iupdate_sxp_dn, ihse_PBE_scf, icheck_dsxH
        end if
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        ihse_mask=0
        do i=1,ntype
            !temp_char = "HSEMASK_PSP"//char(48+i)
            write(temp_char,*) i
            temp_char="HSEMASK_PSP"//ADJUSTL(trim(temp_char))
            temp_char = ADJUSTL(temp_char)
            CALL read_key_words ( 9, TRIM(temp_char),len_trim(temp_char), right, IERR )
            if(ierr.eq.0) then
                ihse_mask=1
                READ ( right, * ) hse_mask_ampl(i), hse_mask_size(i)

                if(inode_tot .eq. 1) then
                    write(22,*) TRIM(temp_char)//"   = ",hse_mask_ampl(i),hse_mask_size(i)
                endif

            else
                hse_mask_ampl(i)=0.d0
                hse_mask_size(i)=1.d0
            endif
        enddo
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, "RELAX_HSE", LEN('RELAX_HSE'), right, IERR )
        if(ierr.eq.0) then
            !ccccc note: we are no longer use
            !num_mov_hse,linemin_hse_relax,irelax_hseLDA (they
            !are used in atomic_relaxation_hseldaCG.f90, but this code is no longer
            !used, APril 1, 2017, LWW
            READ ( right, * ) num_mov_LDA,fac_HSELDA,ihse_relax_lda
        else
            num_mov_LDA=20
            fac_HSELDA = 0.05
            ihse_relax_lda=2
        endif

        if(inode_tot.eq.1) then
            write(22,1125) num_mov_LDA,fac_HSELDA,ihse_relax_lda
        endif
        1125     format(" RELAX_HSE    =",2x,i4,2x,E12.5,2x,i4)


        CALL read_key_words ( 9, "VDW", LEN("VDW"), right, ierr )
        if ( ierr == 0 ) then
            read ( right, '(A30)' ) vdw_method
            call capital(vdw_method, len(vdw_method))
            select case(trim(vdw_method))
            case ("DFT-D2")
                has_vdw = .true.
                has_london = .true.
                has_vdwpair = .false.
                ldftd3 = .false.
            case ("DFT-D3")
                has_vdw = .true.
                has_london = .false.
                ldftd3 = .true.
                has_vdwpair = .false.
            case ("PAIR")
                has_vdw= .true.
                has_vdwpair= .true.
                has_london = .false.
            case default
                has_vdw = .false.
                has_london = .false.
                ldftd3 = .false.
                has_vdwpair = .false.
                vdw_method="NONE"
            end select
            if ( has_london ) then
                call  read_key_words ( 9, "LONDON_S6", len('LONDON_S6'), right, ierr )
                if ( ierr == 0 ) then
                    read ( right , * ) london_s6
                else
                    london_s6=0.75d0
                endif
                do ia = 1, ntype
                    write ( temp_char, * ) ia
                    temp_char = adjustl(temp_char)
                    tmp_char = 'LONDON_C6('//trim(temp_char)//')'
                    call read_key_words ( 9, trim(tmp_char), len_trim(tmp_char), right, ierr )
                    if ( ierr == 0 ) read ( right, * ) london_c6(ia)
                end do
                call read_key_words ( 9, 'LONDON_RCUT', len('LONDON_RCUT'), right, ierr )
                if ( ierr == 0 ) then
                    read ( right , * ) london_rcut
                else
                    london_rcut=200.d0
                endif

            end if
            if (ldftd3) then
                call  read_key_words ( 9, "DFTD3_S6", len('DFTD3_S6'), right, ierr )
                if ( ierr == 0 ) read ( right , * ) dftd3_s6
                call  read_key_words ( 9, "DFTD3_RS6", len('DFTD3_RS6'), right, ierr )
                if ( ierr == 0 ) read ( right , * ) dftd3_rs6
                call  read_key_words ( 9, "DFTD3_S18", len('DFTD3_S18'), right, ierr )
                if ( ierr == 0 ) read ( right , * ) dftd3_s18
                call  read_key_words ( 9, "DFTD3_RS18", len('DFTD3_RS18'), right, ierr )
                if ( ierr == 0 ) read ( right , * ) dftd3_rs18
                call  read_key_words ( 9, "DFTD3_ALPHA6", len('DFTD3_ALPHA6'), right, ierr )
                if ( ierr == 0 ) read ( right , * ) dftd3_alp
                call  read_key_words ( 9, "DFTD3_VERSION", len('DFTD3_VERSION'), right, ierr )
                if ( ierr == 0 ) read ( right , * ) dftd3_version
                call  read_key_words ( 9, "DFTD3_3BODY", len('DFTD3_3BODY'), right, ierr )
                if ( ierr == 0 ) read ( right , * ) dftd3_threebody
                pars(1) = dftd3_s6
                pars(2) = dftd3_rs6
                pars(3) = dftd3_s18
                pars(4) = dftd3_rs18
                pars(5) = dftd3_alp
                call dftd3_set_params(dftd3,pars,dftd3_version)
                dftd3_in%threebody = dftd3_threebody
                CALL dftd3_init(dftd3, dftd3_in)
                if (inode_tot == 1) CALL dftd3_printout(dftd3, dftd3_in)
                CALL dftd3_set_functional(dftd3, func=get_dft_short,version=dftd3_version,tz=.false.)
            end if
        else
            has_vdw = .false.
            has_london = .false.
            ldftd3 = .false.
            vdw_method="NONE"
        end if
        if(inode_tot.eq.1) then
            write(22,*) "VDW       = ", trim(vdw_method)
            write(22,*) "LONDON_S6      = ", london_s6
            do ia = 1, ntype
                write ( temp_char, * ) ia
                temp_char = adjustl(temp_char)
                tmp_char = 'LONDON_C6('//trim(temp_char)//')'
                write(22,*) trim(tmp_char)//"      = ", london_c6(ia)
            end do
            write(22,*) "LONDON_RCUT      = ", london_rcut
            write (22, *) "DFTD3_S6        = ", dftd3_s6
            write (22, *) "DFTD3_RS6        = ", dftd3_rs6
            write (22, *) "DFTD3_S18        = ", dftd3_s18
            write (22, *) "DFTD3_RS18        = ", dftd3_rs18
            write (22, *) "DFTD3_ALPHA6        = ", dftd3_alp
            write (22, *) "DFTD3_VERSION        = ", dftd3_version
            write (22, *) "DFTD3_3BODY        = ", dftd3_threebody
        end if
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !          CALL read_key_words ( 9, "SMTH", LEN('SMTH'), right, IERR )
        !          if(ierr.eq.0) then
        !              READ ( right, * ) Smth
        !          else
        Smth=1.d0
        !          endif

        !          if(inode_tot.eq.1) then
        !              write(22,*) "SMTH      = ", Smth
        !          endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
 !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


        if(is_SOM.eq.1) then
          if(xctype(1).eq.100.and.xctype(2).eq.1.and.xctype(3).eq.9) then ! LDA
            nint_rho_diag_fix=0   ! default for LDA
          else
            nint_rho_diag_fix=1000  ! default for everything else
          endif

            CALL read_key_words ( 9, "SPIN222_MAGDIR_STEPFIX", LEN('SPIN222_MAGDIR_STEPFIX'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, * ) nint_rho_diag_fix
            endif
            if (inode_tot.eq.1) write(22,*) "SPIN222_MAGDIR_STEPFIX=",nint_rho_diag_fix
        endif

 !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            iflag_SCF_neq=0
            CALL read_key_words(9,'SCF_SPECIAL',LEN('SCF_SPECIAL'),right,IERR)
            if(ierr.eq.0) then
              read(right, *)    &
              iflag_SCF_neq,Ef_neq,i1_neq,i2_neq,j1_neq,j2_neq,k1_neq,k2_neq
              Ef_neq=Ef_neq/27.211396d0
              WRITE (22, "(' SCF_SPECIAL   = ', i2, 2x, E13.6, 3(i5,1x,i5,3x))") &
              iflag_SCF_neq,   &
               Ef_neq*27.211396d0,i1_neq,i2_neq,j1_neq,j2_neq,k1_neq,k2_neq
            else
             iflag_SCF_neq=0
            endif
 !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc



        CALL read_key_words ( 9, "COULOMB", LEN('COULOMB'), right, IERR )
        if(ierr.eq.0) then
            !READ ( right, * ) icoul_888, xcoul_888(1), xcoul_888(2),
            !+                       xcoul_888(3)
            READ (right, *) icoul_888
            SELECT CASE (icoul_888)
            CASE (1)
                READ (right, *) icoul_888, xcoul_888(1), xcoul_888(2),xcoul_888(3)
                IF (inode_tot == 1) THEN
                    WRITE (22, "(' COULOMB   = ', i2,3(1x,f10.5))") icoul_888, xcoul_888(1), xcoul_888(2), xcoul_888(3)
                ENDIF
            CASE (11)
                READ (right, *) icoul_888, xcoul_888(1)
                xcoul_888(2)=0.d0
                xcoul_888(3)=0.d0
                IF (inode_tot == 1) THEN
                    WRITE (22, "(' COULOMB   = ', i2, 2x, f10.5)") icoul_888,xcoul_888(1)
                ENDIF
            CASE (12)
                READ (right, *) icoul_888, xcoul_888(2)
                xcoul_888(1)=0.d0
                xcoul_888(3)=0.d0
                IF (inode_tot == 1) THEN
                    WRITE (22, "(' COULOMB   = ', i2, 2x, f10.5)") icoul_888,xcoul_888(2)
                ENDIF
            CASE (13)
                READ (right, *) icoul_888, xcoul_888(3)
                xcoul_888(1)=0.d0
                xcoul_888(2)=0.d0
                IF (inode_tot == 1) THEN
                    WRITE (22, "(' COULOMB   = ', i2, 2x, f10.5)") icoul_888,xcoul_888(3)
                ENDIF
            CASE DEFAULT
                icoul_888 = 0
                xcoul_888(1)=0.d0
                xcoul_888(2)=0.d0
                xcoul_888(3)=0.d0
                IF (inode_tot == 1) THEN
                    WRITE (22, "(' COULOMB   = ', i2)") icoul_888
                ENDIF
            END SELECT
            if(xcoul_888(1)<0.d0 .or. xcoul_888(2)<0.d0 .or. xcoul_888(3)<0.d0) then
                write(message,*) "COULOMB=[0,1,11,12,13] {X1 {X2 {X3}}}, X1,X2,X3 should be in [0,1]"
                call error_stop(message,__FILE__,__LINE__)
            endif
            xcoul_888=mod(xcoul_888,1.d0)
            ! 
            icoul_888 = icoul_888 * 1.001
            !
        else
            icoul_888 = 0 
            xcoul_888(1)=0.d0
            xcoul_888(2)=0.d0
            xcoul_888(3)=0.d0
            IF (inode_tot == 1) THEN
                WRITE (22, "(' COULOMB   = ', i4)") icoul_888
            ENDIF
        endif

        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !ccccc    We need to modify RDATAB (drdatab.f), so it can read in multiple string
        !ccccc    separated by space, or other parser, LWW
        !cccccc    right now, we assume that the second wavefunction is named: (firstname)_2
        !ccccccccc NEED TO BE CHANGED
        !
        !
        CALL read_key_words ( 9, 'IN.WG', LEN('IN.WG'), right, IERR )
        if(ierr.eq.0) then
            read ( right, * ) right_logical
            if ( right_logical ) then
                iwg_in = 1
                fwg_in(1)='IN.WG'
                if(islda.eq.2) then
                    fwg_in(2)=trim(fwg_in(1))//"_2"
                endif
                if(inode_tot.eq.1) then
                    write(22,*) "IN.WG     = ", right_logical
                endif
            else
              if(JOB_1.eq."SCFEP".or.iflag_hpsi.eq.1) then
                    if(inode_tot.eq.1) then
                        write(message,*) "JOB=SCFEP or HPSI, must have IN.WG"
                        call error_stop(message,__FILE__,__LINE__)
                    end if
                endif

                iwg_in = 0
                if(inode_tot.eq.1) then
                    write(22,*) "IN.WG     = ", right_logical
                endif
            endif
        else
          if(JOB_1.eq."SCFEF".or.iflag_hpsi.eq.1) then
                if(inode_tot.eq.1) then
                    write(message,*) "JOB=SCFEF or HPSI, must have IN.WG"
                    call error_stop(message,__FILE__,__LINE__)
                end if
            endif
            if(trim(JOB).eq."DOS") then
                iwg_in = 1
                fwg_in(1)='IN.WG'
                if(islda.eq.2) then
                    fwg_in(2)=trim(fwg_in(1))//"_2"
                endif
                if(inode_tot.eq.1) then
                    write(22,*) "IN.WG     = ", .TRUE.
                endif
            else
                iwg_in = 0
                if(inode_tot.eq.1) then
                    write(22,*) "IN.WG     = ", .FALSE.
                endif
            endif
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if(PWSCF_OUTPUT.and.((JOB_1.eq."SCF").or.(JOB_1.eq."NONSCF"))) then !pwscf
            IWG_OUT = 1
            fwg_out(1) = 'OUT.WG'
            if(islda.eq.2) then
                fwg_out(2)=trim(fwg_out(1))//"_2"
            endif
            right_logical = .TRUE.
            WRITE (temp_char, *) right_logical
            if(inode_tot.eq.1) write(22,*) "OUT.WG    = "//TRIM(ADJUSTL(temp_char))
        else   !pwscf
            CALL read_key_words ( 9, 'OUT.WG', LEN('OUT.WG'), right, IERR )
            if(ierr.eq.0) then
                read ( right, * ) right_logical
                IF ( right_logical ) THEN
                    IWG_OUT = 1
                    fwg_out(1) = 'OUT.WG'
                    if(islda.eq.2) then
                        fwg_out(2)=trim(fwg_out(1))//"_2"
                    endif
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.WG    = ", right_logical
                    endif
                ELSE
                    IWG_OUT = 0
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.WG    = ", right_logical
                    endif
                ENDIF
            else
                if (trim(JOB_1).eq."SCF" .or. trim(JOB_1)=='NONSCF'.or. trim(JOB_1).eq."NAMD".or.trim(JOB_1).eq."TDDFT".or.trim(JOB_1).eq."RELAX") then
                    IWG_OUT = 1
                    fwg_out(1) = 'OUT.WG'
                    if(islda.eq.2) then
                        fwg_out(2)=trim(fwg_out(1))//"_2"
                    endif
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.WG    = ", .TRUE.
                    endif

                else
                    IWG_OUT = 0
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.WG    = ", .FALSE.
                    endif
                endif
            endif
        endif  ! pwscf
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        CALL read_key_words ( 9, 'IN.RHO', LEN('IN.RHO'), right, IERR )
        if(ierr.eq.0) then 
            read ( right, * ) right_logical
            IF ( right_logical ) THEN
                irho_in   = 1
                frho_in(1)='IN.RHO'
                if(islda.eq.2) then
                    frho_in(2)=trim(frho_in(1))//"_2"
                endif
                if(inode_tot.eq.1) then
                    write(22,*) "IN.RHO    = ", right_logical
                endif
            ELSE
                irho_in   = 0
                if(inode_tot.eq.1) then
                    write(22,*) "IN.RHO    = ", right_logical
                endif
            ENDIF
        else
            irho_in   = 0
            if(inode_tot.eq.1) then
                write(22,*) "IN.RHO    = ", .FALSE.
            endif
        endif

        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        iflag_rho_add=0
        rho_add_fact=0
        rho_add_band=0
        CALL read_key_words ( 9, 'IN.RHO_ADD', LEN('IN.RHO_ADD'), right, IERR )
        if(ierr.eq.0) then 
            read ( right, * ) right_logical
            IF ( right_logical ) THEN
              iflag_rho_add=1
              read(right,*,iostat=ierror) right_logical,rho_add_fact,rho_add_band
              if(ierror.ne.0) then
                  read(right,*,iostat=ierror) right_logical,rho_add_fact
                  if(ierror.ne.0) then
                      rho_add_fact=1.d0
                  endif
              endif
                if(inode_tot.eq.1) then
                    write(22,*) "IN.RHO_ADD    = ", right_logical,rho_add_fact,rho_add_band
                endif
            ELSE
              iflag_rho_add=0
                if(inode_tot.eq.1) then
                    write(22,*) "IN.RHO_ADD    = ", right_logical
                endif
            ENDIF
        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        if(is_SOM.eq.1) then
            frho_in_SOM="IN.RHO_SOM";
            frho_out_SOM="OUT.RHO_SOM";
        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        IF(PWSCF_OUTPUT.or.(JOB_1.eq."SCF").or.(JOB_1.eq."NONSCF").or.(JOB_1.eq."RELAX")) THEN !pwscf
            irho_out = 1
            frho_out(1) = 'OUT.RHO'
            if(islda.eq.2) then
                frho_out(2)=trim(frho_out(1))//"_2"
            endif
            right_logical = .TRUE.
            WRITE (temp_char, *) right_logical
            if(inode_tot.eq.1) write(22,*) "OUT.RHO   = "//TRIM(ADJUSTL(temp_char))
        ELSE   !pwscf

            CALL read_key_words ( 9, 'OUT.RHO', LEN('OUT.RHO'),right, IERR )
            if(ierr.eq.0) then
                read ( right, * ) right_logical
                IF ( right_logical ) THEN
                    irho_out = 1
                    frho_out(1) = 'OUT.RHO'
                    if(islda.eq.2) then
                        frho_out(2)=trim(frho_out(1))//"_2"
                    endif
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.RHO   = ", right_logical
                    endif
                ELSE
                    irho_out = 0 
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.RHO   = ", right_logical
                    endif
                ENDIF
            else
                if(trim(JOB_1).eq."SCF".or.trim(JOB_1).eq."NAMD".or.trim(JOB_1).eq."TDDFT") then
                    irho_out = 1
                    frho_out(1) = 'OUT.RHO'
                    if(islda.eq.2) then
                        frho_out(2)=trim(frho_out(1))//"_2"
                    endif
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.RHO   = ", .TRUE.
                    endif
                else
                    irho_out = 0
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.RHO   = ", .FALSE.
                    endif
                endif
            endif

        ENDIF  !pwscf
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9 , 'IN.VR', LEN('IN.VR'), right, IERR )
        if(ierr.eq.0) then
            read ( right, * ) right_logical
            IF ( right_logical ) THEN
                ivr_in = 1
                fvr_in(1) = 'IN.VR'
                if(islda.eq.2) then
                    fvr_in(2)=trim(fvr_in(1))//"_2"
                endif
                if(inode_tot.eq.1) then
                    write(22,*) "IN.VR     = ", right_logical
                endif
            ELSE
                ivr_in = 0 
                if(inode_tot.eq.1) then
                    write(22,*) "IN.VR     = ", right_logical
                endif
            ENDIF
        else
            ivr_in=0
            if(inode_tot.eq.1) then
                write(22,*) "IN.VR     = ", .FALSE.
            endif         
        endif
        if (JOB_1 == 'NONSCF' .and. ivr_in == 0 .and. inode_tot == 1) then
            write (6, *) 'For JOB=NONSCF please set IN.VR = T in etot.input'
            call mpi_abort(mpi_comm_world,ierr)
        end if
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'OUT.VR', LEN('OUT.VR'), right, IERR )
        if(ierr.eq.0) then
            read ( right, * ) right_logical
            IF ( right_logical ) THEN
                ivr_out = 1
                fvr_out(1) = 'OUT.VR'
                if(islda.eq.2) then
                    fvr_out(2)=trim(fvr_out(1))//"_2"
                endif
                if(inode_tot.eq.1) then
                    write(22,*) "OUT.VR    = ", right_logical
                endif
            ELSE
                ivr_out = 0
                if(inode_tot.eq.1) then
                    write(22,*) "OUT.VR    = ", right_logical
                endif
            ENDIF
        else
            if(trim(JOB_1).eq."SCF".or.trim(JOB_1).eq."NAMD".or.trim(JOB_1).eq."TDDFT".or.trim(JOB_1).eq."RELAX") then
                ivr_out = 1
                fvr_out(1) = 'OUT.VR'
                if(islda.eq.2) then
                    fvr_out(2)=trim(fvr_out(1))//"_2"
                endif
                if(inode_tot.eq.1) then
                    write(22,*) "OUT.VR    = ", .TRUE.
                endif
            else
                ivr_out=0
                if(inode_tot.eq.1) then
                    write(22,*) "OUT.VR    = ", .FALSE.
                endif
            endif
        endif

        if(trim(JOB_store).eq."POTENTIAL") then
            ivr_out=2  
        endif

        ! just calculate  the potential, then output OUT.VR, OUT.VR_hion
        ! then stop, the JOB has been set to SCF
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'IN.VEXT', LEN('IN.VEXT'),right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) right_logical
            IF ( right_logical ) THEN
                ivext_in = 1
                fvext_in = 'IN.VEXT'
                if(inode_tot.eq.1) then
                    write(22,*) "IN.VEXT   = ", right_logical
                endif
            ELSE
                ivext_in = 0
                if(inode_tot.eq.1) then
                    write(22,*) "IN.VEXT   = ", right_logical
                endif
            ENDIF
        else
            ivext_in =0
            if(inode_tot.eq.1) then
                write(22,*) "IN.VEXT   = ", .FALSE.
            endif
        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'OUT.VATOM', LEN('OUT.VATOM'),right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) right_logical
            if(inode_tot.eq.1) then
                write(22,*) "OUT.VATOM   = ", right_logical
            endif
            IF ( right_logical ) THEN
                iout_vatom = 1
            ELSE
                iout_vatom = 0
            ENDIF
        else
            iout_vatom =0
            if(inode_tot.eq.1) then
                write(22,*) "OUT.VATOM   = ", .FALSE.
            endif
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        

        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'PLUGIN_SPECIAL1',LEN('PLUGIN_SPECIAL1'),right, IERR )

        if(ierr.eq.0) then
            read(right,*) right_logical
            if(right_logical) then
                iflag_plugin_special1=1
                READ(right,*,IOSTAT=ierror) right_logical,plugin_param(1)
                if(ierror.ne.0) then
                    if(inode_tot.eq.1) then
                        write(message,*) "PLUGIN_SPECIAL1 must has one parameter"
                        call error_stop(message,__FILE__,__LINE__)
                    end if
                endif

                if(inode_tot.eq.1) then
                    write(22,*) "PLUGIN_SPECIAL1 = ", .TRUE., plugin_param(1)
                endif

            else
                iflag_plugin_special1=0
            endif
        else
            iflag_plugin_special1=0
        endif


        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        IF ( TRIM( JOB ) .eq. "RELAX" .OR. TRIM(JOB) .eq. "MD" .OR. TRIM(JOB).eq."SCFEP" ) THEN      

            IFORCE = 1
            fforce_out = "OUT.FORCE"
            IF ( inode_tot .eq. 1 ) THEN
                write ( 22, * ) "OUT.FORCE = ", .TRUE.
            ENDIF
        ELSE
            CALL read_key_words ( 9, 'OUT.FORCE',LEN('OUT.FORCE'),right, IERR )
            if ( ierr .eq. 0 ) then
                READ ( right, * ) right_logical 
                IF ( right_logical ) THEN
                    IFORCE = 1
                    fforce_out = 'OUT.FORCE'
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.FORCE = ", right_logical
                    endif
                ELSE
                    IFORCE = 0
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.FORCE = ", right_logical
                    endif
                ENDIF
            else
                IFORCE = 0
                if(inode_tot.eq.1) then
                    write(22,*) "OUT.FORCE = ", .FALSE.
                endif
            endif
        ENDIF

        fstress_out = 'OUT.STRESS'
        IF ( TRIM( JOB ) .eq. "RELAX") THEN
            if(istress_cal .eq. 1) then
                if(inode_tot.eq.1) then
                    write(22,*) "OUT.STRESS = ", .TRUE.
                endif
            else
                if(inode_tot.eq.1) then
                    write(22,*) "OUT.STRESS = ", .FALSE.
                endif
            endif
        ENDIF

        IF ( TRIM( JOB ) .ne. "RELAX") THEN
            CALL read_key_words ( 9, 'OUT.STRESS',LEN('OUT.STRESS'),right, IERR )
            if ( ierr .eq. 0 ) then
                READ ( right, * ) right_logical 
                IF ( right_logical ) THEN
                    istress_cal = 1
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.STRESS = ", right_logical
                    endif
                ELSE
                    istress_cal = 0
                    if(inode_tot.eq.1) then
                        write(22,*) "OUT.STRESS = ", right_logical
                    endif
                ENDIF
            else
                istress_cal = 0
                if(inode_tot.eq.1) then
                    write(22,*) "OUT.STRESS = ", .FALSE.
                endif
            endif
        ENDIF
        MCTRL_stress=istress_cal

        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        if(gen_from_mp .eq. 0) then
            CALL read_key_words ( 9, 'IN.SYMM', LEN('IN.SYMM'), right, IERR)
            if(ierr.eq.0) then
                READ ( right, * ) right_logical
                IF ( right_logical ) THEN
                    ISYM = 1
                    sym_file = 'IN.SYMM'
                ELSE
                    ISYM = 0
                ENDIF
            else
                ISYM = 0
            endif
        else
            mpn123_nosymm=.false.
            CALL read_key_words ( 9, 'IN.SYMM', LEN('IN.SYMM'), right, IERR)
            if(ierr.eq.0) then
                READ ( right, * ) right_logical
                IF ( right_logical ) THEN
                    ISYM = 1
                ELSE
                    ISYM = 0
                    mpn123_nosymm=.true.
                ENDIF
            else
                ISYM = 1
            endif
        endif
        if(ISYM .eq. 0) then
            if(inode_tot.eq.1) then
                write(22,*) "IN.SYMM   = ", .FALSE.
            endif
        else
            if(inode_tot.eq.1) then
                write(22,*) "IN.SYMM   = ", .TRUE.
            endif
        endif
        if(gen_from_mp .eq. 1) then
            !! over write the IN.SYMM
            ISYM = 1
            sym_file = 'OUT.SYMM'
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        ! Weile Jia, for HSE.  HSE HSE
        if(use_hse .eq. 1) then
            use_sx_in = .false.
            if(ISYM /= 0) then
                use_symm_op_1 = .true.
            else
                use_symm_op_1 = .false.
            endif
            use_symm_op_2 = .false.
            use_kpt_sx = .false.
            sx_mix_rate(:) = hse_alpha
            if(use_hse .eq. 1.and. num_group_k .gt. 1) then
                if(inode_tot.eq.1) then
                    write(message,*) "HSE can't use K-parallel"
                    call error_stop(message,__FILE__,__LINE__)
                end if
                !           stop
            endif
            !         write(*,*) '***********************************************'
            !         write(*,*) '**** Running HSE program... it is not over*****'
            !         write(*,*) '***********************************************'
            !         write(*,*) '**** HSE  HSE HSE HSE HSE HSE HSE BEGIN..******'
            !         write(*,*) '***********************************************'
            !         write(*,*) 'use_symm_op_1 is', use_symm_op_1
            !         write(*,*) 'use_symm_op_2 is', use_symm_op_2
            !         write(*,*) 'hse flag', use_hse
            !         write(*,*) 'sx mix rate:', sx_mix_rate(1:2)
            !         write(*,*) '***********************************************'
            sx_para_g = nnodes_b
            sx_para_k = num_group_k

        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


        if(gen_from_mp .eq. 0) then
            CALL read_key_words ( 9, 'IN.KPT', LEN('IN.KPT'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, * ) right_logical 
                IF ( right_logical ) THEN
                    ikpt_yno_999 =1
                    kpt_file = 'IN.KPT'
                    if(inode_tot.eq.1) then
                        write(22,*) "IN.KPT    = ", right_logical
                    endif
                ELSE
                    ikpt_yno_999 = 0
                    if(inode_tot.eq.1) then
                        write(22,*) "IN.KPT    = ", right_logical
                    endif
                ENDIF
            else
                ikpt_yno_999 = 0
                if(inode_tot.eq.1) then
                    write(22,*) "IN.KPT    = ", .FALSE.
                endif
            endif
        else
            !! over write the IN.KPT
            ikpt_yno_999 = 1
            kpt_file = 'OUT.KPT'
        endif 

        if(natom.gt.0) then
            if(inode_tot .eq. 1) then

                allocate(ityp_old(natom))
                do i=1,natom
                    ityp_old(i)=ityp(i)
                    ityp(i)=ityp(i)+weight_mag(i)*1000   ! just in case there are magnetic input
                enddo

                if(gen_from_mp==1) then
                    call check(n1l, n2l, n3l)  ! this is to generate the symmetry and reduced kpoints
                endif

                do i=1,natom
                    ityp(i)=ityp_old(i)
                enddo
                deallocate(ityp_old)

            endif
            call mpi_barrier(mpi_comm_world, ierr)
        endif

        call readkpt()    
        if (inode_tot.eq.1) write(22,*) "NUM_KPT = ", nkpt


        !!!!!!!!!! energy decomposition
        rad_box=12.d0   ! this will be used in many other functions(e.g.,solvent)
        !!! We might need a input parameter. If there is no atom within this
        !rad_box, then the decomposition might not be good
        call read_key_words (9, 'CHARGE_DECOMP',LEN('CHARGE_DECOMP'),right,IERR)
        if(ierr == 0) then
            READ( right,*) right_logical
            if(right_logical) then
                iflag_chargedecomp=1
            else
                iflag_chargedecomp=0
            endif
        else
            iflag_chargedecomp=0
        endif
        if(inode_tot.eq.1) then
            if(iflag_chargedecomp==1) then
                write(22,*) "CHARGE_DECOMP = T"
            else
                write(22,*) "CHARGE_DECOMP = F"
            endif
        endif
        !!!!!!!!!! energy decomposition
        ! this should always be zero, only assigned to 1 before calling
        ! Hpsi_comp_AllBandBP2_float, and 
        ! assigned back to zero after call Hpsi_comp_AllBandBP2_float
        ! This parameter is passed into Hpsi_comp_AllBandBP2_float inside
        ! mod_energydecomp1
        call read_key_words (9, 'ENERGY_DECOMP',LEN('ENERGY_DECOMP'),right,IERR)
        if (ierr == 0) then
          READ ( right, *, IOSTAT=ierror ) right_logical,iflag_energydecomp_type
            if(ierror.ne.0) then
            READ ( right, *) right_logical
            iflag_energydecomp_type=1
            endif

              if(iflag_energydecomp_type.eq.11.or.iflag_energydecomp_type.eq.22) then
              allocate(weight_decomp(natom))
              allocate(dQ_decomp(natom,natom))
              allocate(dW_decomp(natom,natom))
              num_dW=0
              iposit_dW=0
              weight_decomp=1.d0
              endif

            if(right_logical) then
                iflag_energydecomp=1
                if(precision_flag.ne.double_prec) then
                    if(inode_tot.eq.1) then
                        !  write(message,*) "ENERGY_DECOMP=T, MUST HAVE PRECISION=DOUBLE"
                        !   call error_stop(message,__FILE__,__LINE__)
                        write(22,*) "IT MIGHT BE BETTER TO USE PRECISION=DOUBLE FOR ENERGY_DECOMP=T"
                        write(6,*) "IT MIGHT BE BETTER TO USE PRECISION=DOUBLE FOR ENERGY_DECOMP=T"
                    end if
                endif 


        !        READ(right,*,IOSTAT=ierror) right_logical,(eta_fact(ii),ii=1,ntype)
        !        if(ierror.ne.0) then
        !            eta_fact(1:ntype)=1.d0
        !        endif
            else
                iflag_energydecomp=0
            endif
        else
            iflag_energydecomp=0
        endif



        if(inode_tot.eq.1) then
            if(iflag_energydecomp==1) then
              write(22,*) "ENERGY_DECOMP = T ", iflag_energydecomp_type
            else
                write(22,*) "ENERGY_DECOMP = F "
            endif
        endif
        if(iflag_energydecomp.eq.1 .and. num_group_k .gt. 1) then
            if(inode_tot.eq.1) then
                write(6,*) "num_group_k=", num_group_k
                write(message,*) "ENERGY_DECOMP=T, CAN NOT USE K-POINTS PARALLELIZATION"
                call error_stop(message,__FILE__,__LINE__)
            end if
        endif
        if(iflag_energydecomp.eq.1 .and. (n1.ne.n1L .or. n2.ne.n2L .or. n3 .ne. n3L)) then
            if(inode_tot.eq.1) then
                write(message,*) "ENERGY_DECOMP=T, SHOULD N123==N123L"
                call error_stop(message,__FILE__,__LINE__)
            end if
        endif
        !      if(iflag_energydecomp.eq.1 .and. nkpt .gt. 1) then
        !           if(inode_tot.eq.1) then
        !           write(message,*) "ENERGY_DECOMP=T, CAN NOT USE MULTI K-POINTS"
        !           call error_stop(message,__FILE__,__LINE__)
        !           end if
        !      endif


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!        This is very confusion. Here are the choices and logic
!        The main logic flow parameters are: 
!        iflag_energydecomp_type (called it type) in line: ENERGY_DECOMP = T, type
!        iflag_natom_coulomb_method (call it mthd) in line: ENERGY_DECOMP_COULOMB
!        iflag_natom_coulomb (call it Coulomb) in(with/without) line: ENERGY_DECOMP_COULOMB (T or F)
!        --------------------------------------------------------
!        type=1, or 2  : call gen_motall, fixed weight Hirshfeld
!        type=11 or 22 : call gen_motall2, adjust weight w(iat) Hirshfeld, each atom is charge neutral
!        --------------------------------------
!        Coulomb=F (no coulomb):
!        type=1, or 11 : call double_vion, 1/8pi*|grad_V|^2 for partitioning
!        type=2, or 22 : call double_vion2,  0.5*(Vcoul(r)*(rho_el(r)-rho_nucl(r)) for partitioning
!        ----------------------------------------
!        Coulomb=T (with Coulomb fitting), then type=1/2, the same, type=11/22 the same
!         mthd=1, (do simple Gaussian fitting, call double_vion_coulomb.f90) 
!                     1/8pi*|grad_Vres|^2 + Vres(r)*rho_fit(r) for partitioning
!                                             Vres(r)*rho_fit_atom already calculated for each atom, so not
!                                             participated in Hirshfeld partitioning (show as one column in OUT.ENDIV)
!        mthd=2, (do simple Guassian fitting, call double_vion_coulomb2.f90) 
!                     1/8pi*[|grad_V|^2*fac1-|grad_Vfit|^2]*fac1 for partitioning
!                                             fac1,fac2 are input in the ENERGY_DECOMP_COULOMB line, used
!                                             here for analysis
!       mthd=3, (use funcq_atom.fit, numerical fitting, call double_vion_coulomb3.f90) 
!                     1/8pi*|grad_Vres|^2+Vres(r)*rho_fit(r)
!        ------------------------------------------
!        Recommend: use type 1, for Coulomb=T, use mthd=3
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       
        

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        iflag_natom_coulomb=0
        call read_key_words (9, 'ENERGY_DECOMP_COULOMB',LEN('ENERGY_DECOMP_COULOMB'),right,IERR)
        if (ierr == 0) then
          READ ( right, * )   &
          right_logical,iflag_natom_neutral,iflag_natom_coulomb_method,frac_coulomb_res1, &
          frac_coulomb_res2,num_G_coul,q2_fit,iflag_output_CoulPot
            if(right_logical) then
                iflag_natom_coulomb=1
                write(22,*) "ENERGY_DECOMP_COULOMB = T", &
                 iflag_natom_neutral,iflag_natom_coulomb_method,frac_coulomb_res1,frac_coulomb_res2
                if(num_G_coul.ne.1.and.num_G_coul.ne.2) then
                 write(6,*) "the number of Gausian in ENERGY_DECOMP_COULOMB must be 1 or 2, stop"
                 stop
                endif
            else
                iflag_natom_coulomb=0
                write(22,*) "ENERGY_DECOMP_COULOMB = F ",iflag_natom_neutral, &
                    iflag_natom_coulomb_method,frac_coulomb_res1,frac_coulomb_res2,num_G_coul,q2_fit
            endif
            q2_fit=(q2_fit/0.529177)**2/4
        endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

                if(iflag_natom_coulomb.eq.1) then
                  if(iflag_natom_neutral.eq.1) then

                  open(24,file="IN.CONSTRAINT_COULOMB")
                  rewind(24)
                  read(24,*) N_constraint_coulomb

!cccccccccccc  we should already read in natom_coulomb
                  allocate(ind_constraint_charge(natom_coulomb,N_constraint_coulomb))
                  allocate(num_constraint_charge(N_constraint_coulomb))
                  allocate(q_constraint_charge(N_constraint_coulomb))
                  do ii=1,N_constraint_coulomb
                  read(24,*) num,q,(ind_constraint_charge(j,ii),j=1,num)
                  num_constraint_charge(ii)=num
                  q_constraint_charge(ii)=q
                  enddo
                  close(24)
                else
                   N_constraint_coulomb=0
                 endif
               endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
          if(iflag_natom_coulomb_method.eq.3) then
           open(12,file="funcq_atom.fit") 
           rewind(12)
           read(12,*) ntype_tmp
           if(ntype_tmp.ne.ntype) then
           write(6,*) "ntype in funcq_atom.fit ne ntype in etot.input,stop"
           stop
           endif
           allocate(itmp(ntype))
           read(12,*) (itmp(ii),ii=1,ntype)
           read(12,*) mnq_coul
           allocate(rhoq_coul(mnq_coul,ntype))
           allocate(rhoq_tmp(mnq_coul,ntype))
           allocate(qi_coul(mnq_coul))
           do iq=1,mnq_coul
           read(12,*) qi_coul(iq),(rhoq_tmp(iq,ii),ii=1,ntype)
           enddo
           close(12)
           do itype=1,ntype
            do ii=1,ntype
            if(iiatom(itype).eq.itmp(ii)) then
            do iq=1,mnq_coul
            rhoq_coul(iq,itype)=rhoq_tmp(iq,ii)
            enddo
            endif
            enddo
           enddo
           deallocate(itmp)
           deallocate(rhoq_tmp)
           endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc



        dfftp%nr1 = n3L        ! the 1,2,3, is reversed inpwscf.solvent model, not used in rVV10 !
        dfftp%nr2 = n2L
        dfftp%nr3 = n1L
        dfftp%nr1x = n3L
        dfftp%nr2x = n2L
        dfftp%nr3x = n1L
        dfftp%nnr = n1L*n2L*n3L/nnodes

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! solvent0 MODULE
        CALL read_key_words ( 9, "IN.SOLVENT", LEN('IN.SOLVENT'), right, IERR )
        if (ierr == 0) then
            read (right, *) use_solvent0
        else
            use_solvent0 = .false.
        end if
        if (inode_tot == 1) then
            write (22, *) 'IN.SOLVENT = ', use_solvent0
        end if
        if ( use_solvent0 ) then
            nspin = ispin_input
            IF ( .NOT. ALLOCATED (param_charge) ) ALLOCATE (param_charge(3,sum_atom_type) )
            IF ( .NOT. ALLOCATED (atomicspread) ) ALLOCATE (atomicspread(sum_atom_type) )
            dfftp%nr1 = n3L        ! the 1,2,3, is reversed inpwscf.solvent model
            dfftp%nr2 = n2L
            dfftp%nr3 = n1L
            dfftp%nr1x = n3L
            dfftp%nr2x = n2L
            dfftp%nr3x = n1L
            dfftp%nnr = n1L*n2L*n3L/nnodes
            !call ggen(ecut2,n1,n2,n3)
            CALL solvent0_defaults ()
            CALL read_in_solvent ( sum_atom_type )
            CALL solvent0_base_init ( assume_isolated, solvent0_restart,verbose, solvent0_thr, solvent0_nskip,solvent0_type, stype, rhomax, rhomin, tbeta,env_static_permittivity,env_optical_permittivity, eps_mode,param_charge(1:3,1:sum_atom_type), atomicspread(1:sum_atom_type),add_jellium, ifdtype, nfdpoint,mixtype, ndiis, mixrhopol, tolrhopol,env_surface_tension, delta,env_pressure,env_ioncc_level, nrep, cion, zion, rhopb,solvent_temperature,env_external_charges, extcharge_charge,extcharge_dim, extcharge_axis,extcharge_pos, extcharge_spread,env_dielectric_regions, epsregion_eps,epsregion_dim, epsregion_axis,epsregion_pos, epsregion_spread,epsregion_width )
            CALL solvent0_initions_allocate( sum_atom, sum_atom_type )
            CALL solvent0_initbase ( n1L*n2L*n3L/nnodes )
            allocate ( zv(sum_atom_type) )
            do i = 1, sum_atom_type
                zv(i) = upfpsp(i)%zp
            end do
            CALL solvent0_initions ( n1L*n2L*n3L/nnodes, sum_atom, sum_atom_type, ityp, zv, tau, alat )
            CALL solvent0_initcell ( n1L*n2L*n3L/nnodes, n3L, n2L, n1L, 0, omega_bohr, alat, latt_vec )


            CALL read_key_words ( 9, "OUT.SOLVENT_CHARGE", LEN('OUT.SOLVENT_CHARGE'), right, IERR )
            if (ierr == 0) then
                read (right, *) right_logical
                if(right_logical) then
                    iflag_out_solvent_charge=1
                else
                    iflag_out_solvent_charge=0
                endif
            else
                iflag_out_solvent_charge=0
            end if

        end if
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

            iflag_solvent_energydecomp=0
           if(use_solvent0.and.iflag_energydecomp.eq.1) then
            iflag_solvent_energydecomp=1
            allocate(solvent_extr_eden(n1L*n2L*n3L/nnodes))
           endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'NUM_ELECTRON', LEN('NUM_ELECTRON'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) TOTNEL     ! otherwise, use the TOTNEL calculated above
        endif
        if(inode_tot.eq.1) then
            write(22,*) "NUM_ELECTRON =", TOTNEL
        endif

        iflag_Ef_spin=0
        totNel_spin(1)=totNel/2
        totNel_spin(2)=totNel/2

        CALL read_key_words ( 9, 'NUM_ELECTRON_SPIN', LEN('NUM_ELECTRON_SPIN'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) totNel_spin(1),totNel_spin(2)  
            totNel=totNel_spin(1)+totNel_spin(2)
            iflag_Ef_spin=1
            if(inode_tot.eq.1) then
                write(22,*) "NUM_ELECTRON_SPIN =",totNel_spin(1),totNel_spin(2)
                if(islda.eq.1) then
                    write(22,*) "WARRNING: for SPIN=1,NUM_ELECTRON_SPIN CANNOT BE USED"
                endif
            endif
        endif

        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'FIX_FERMI', LEN('FIX_FERMI'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * )right_logical,EFermi_fix_987,amix_Fermi_987,drho_Pulay_987
            if(right_logical) then
            write(22,"('FIX_FERMI =', i4, 3(E13.6,1x))" )   &
              right_logical,EFermi_fix_987,amix_Fermi_987,drho_Pulay_987
                use_fix_Fermi_987=1
                EFermi_fix_987=EFermi_fix_987/27.211396d0
            else
                use_fix_Fermi_987=0
            endif
        else
            use_fix_Fermi_987=0
            EFermi_fix_987=0.d0
            amix_Fermi_987=0.d0
            drho_Pulay_987=0.d0
        endif

        !*************************************************
        !
        CALL read_key_words ( 9, 'IN.NONSCF', LEN('IN.NONSCF'), right,IERR )
        if(ierr.eq.0) then
            READ ( right, * ) right_logical 
            if(right_logical) then
                call read_nonscf()
            else
                call default_nonscf()
            endif
            if(inode_tot.eq.1) then
                write(22,*) "IN.NONSCF    = ", right_logical
            endif
        else
            call default_nonscf()
            if(inode_tot.eq.1) then
                write(22,*) "IN.NONSCF    = ", .false.
            endif
        endif
        if(is_fsm .and. precision_flag .ne. double_prec) then
            write(message,*) "NONSCF_METH != 0, can only run with PRECISION=DOUBLE, stop"
            call error_stop(message,__FILE__,__LINE__)
        endif
        !*************************************************
        CALL read_key_words ( 9, 'NUM_BAND', LEN('NUM_BAND'),right,IERR)
        if(ierr.eq.0) then
            read ( right, * ) mx
        else
            if(is_SO.ne.1) then
                mx = 1.05*TOTNEL/2+10
                if (islda == 2) mx = mx * 1.2
            else
                mx=1.05*TOTNEL+10
            endif
            if(MCTRL_NONSCF_METH.eq.2) then
                write(message,*) "NONSCF_METH=2, need set NUM_BAND"
                call error_stop(message,__FILE__,__LINE__)
            endif
        endif
        if(trim(JOB_store).eq."POTENTIAL") then
            mx=1
        endif
        if(MCTRL_NONSCF_METH.eq.1 .or. MCTRL_NONSCF_METH.eq.-1 .or. MCTRL_NONSCF_METH.eq.3) then
            mx=1
        endif
        if(inode_tot.eq.1) write(22,1115) mx
        1115   format(" NUM_BAND  =",i6)
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'OUT.REAL.RHOWF_SP', LEN('OUT.REAL.RHOWF_SP'), right, IERR )
        if(ierr.eq.0) then
            READ (right, *) flag_dens
            SELECT CASE (flag_dens)
            CASE (0)
                idens_out=0
                IF (inode_tot == 1) THEN
                    WRITE (temp_char, *) flag_dens
                    WRITE (22, *)  'OUT.REAL.RHOWF_SP= '//TRIM(ADJUSTL(temp_char))
                ENDIF
            CASE (1, 11, 12, 2, 21, 22)
                READ ( right, *,iostat=ierr) flag_dens, kpt_dens(1), kpt_dens(2), ispin_dens(1), ispin_dens(2), iw_dens(1), iw_dens(2) 
                if(ierr.ne.0) then
                READ ( right, *,iostat=ierr) flag_dens, kpt_dens(1), kpt_dens(2), ispin_dens(1), ispin_dens(2)
                iw_dens(1)=1
                iw_dens(2)=mx
                endif
                idens_out= flag_dens
                fdens_out = "OUT.REAL.RHOWF_SP"
                IF (inode_tot == 1) THEN
                    write(temp_char, 2002) flag_dens, kpt_dens(1),kpt_dens(2),ispin_dens(1),ispin_dens(2),iw_dens(1),iw_dens(2)
                    2002        format(i3,3(i5,1x,i5,4x)) 
                    WRITE (22, *) 'OUT.REAL.RHOWF_SP= '//TRIM(ADJUSTL(temp_char))
                ENDIF
            CASE default
                WRITE (6, *) ' OUT.REAL.RHOWF_SP = 0/1/11/2/22, ...?'
                STOP
            END SELECT
        ELSE
            idens_out=0
            IF (inode_tot == 1) THEN
                WRITE (temp_char, *) 0
                WRITE (22, *) 'OUT.REAL.RHOWF_SP= '//TRIM(ADJUSTL(temp_char))
            ENDIF
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9,'WG_ERROR', LEN('WG_ERROR'),right,IERR )
        if(ierr.eq.0) then
            READ ( right, * ) TOLUG
        else
            TOLUG = 1.D-4
            if(CONVERGE .eq. "DIFFICULT")  tolug = tolug * 0.5
        endif
        if(inode_tot.eq.1) write(22,*) "WG_ERROR  = ", TOLUG
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'E_error', LEN('E_Error'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) tolE
        else
            tolE = 1.D-7*TOTNEL*Hartree_ev
            if(CONVERGE .eq. "DIFFICULT")  tolE = tolE * 0.01
        endif
        if(inode_tot.eq.1) then
            write(22,*) "E_ERROR   = ", tolE
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'RHO_ERROR', LEN('RHO_ERROR'),right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) tolrho
        else
            !tolrho = 0.4D-3
            tolrho = 0.5D-4
            if(CONVERGE .eq. "DIFFICULT")  tolrho = tolrho * 0.5
        endif
        if(inode_tot.eq.1) then
            write(22,*) "RHO_ERROR = ", tolrho
            if(tolrho.lt.5.D-8) then
                write(22,*) "!WARNING: RHO_ERROR.lt.5.D-8 can damage Pulay_Mix stored_pair"
            endif
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !          CALL read_key_words ( 9, 'W_CG', LEN('W_CG'), right, IERR )
        !          if(ierr.eq.0) then
        !              READ ( right, * ) W_CG
        !          else
        !              W_CG = 0.07d0    ! this is a test result
        !              if(CONVERGE .eq. "DIFFICULT") W_CG=0.d0
        !          endif
        !          if(inode_tot.eq.1) then
        !              write(22,*) "W_CG      = ", W_CG
        !          endif
        CALL read_key_words ( 9, 'RHO_RELATIVE_ERROR', LEN('RHO_RELATIVE_ERROR'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) RHO_RELATIVE_ERROR
        else
            RHO_RELATIVE_ERROR = 0.07d0    ! this is a test result
            if(CONVERGE .eq. "DIFFICULT") RHO_RELATIVE_ERROR=0.d0
        endif
        if(inode_tot.eq.1) then
            write(22,*) "RHO_RELATIVE_ERROR      = ", RHO_RELATIVE_ERROR
        endif

        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !          CALL read_key_words ( 9, 'W_SCF', LEN('W_SCF'), right, IERR )
        !          if(ierr.eq.0) then
        !              READ ( right, * ) W_SCF
        !          else
        !              if (trim(JOB).eq. "MD") then
        !                  W_SCF = 0.02d0       ! MD w_scf, might need to re-consider
        !              elseif (trim(JOB).eq."RELAX") then
        !                  W_SCF = 0.003d0      ! this is a test result
        !              else
        !                  W_SCF = 0.d0
        !              endif
        !              if(CONVERGE .eq. "DIFFICULT") W_SCF = W_SCF*0.05
        !          endif
        !          if(inode_tot.eq.1) then
        !              write(22,*) "W_SCF     = ", W_SCF
        !          endif
        CALL read_key_words ( 9, 'FORCE_RELATIVE_ERROR', LEN('FORCE_RELATIVE_ERROR'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) FORCE_RELATIVE_ERROR
        else
            if (trim(JOB).eq. "MD") then
                !FORCE_RELATIVE_ERROR = 0.02d0       ! MD FORCE_RELATIVE_ERROR, might need to re-consider
                FORCE_RELATIVE_ERROR = 0.d0
            elseif (trim(JOB).eq."RELAX") then
                !FORCE_RELATIVE_ERROR = 0.003d0      ! this is a test result
                FORCE_RELATIVE_ERROR = 0.d0
            else
                FORCE_RELATIVE_ERROR = 0.d0
            endif
            if(CONVERGE .eq. "DIFFICULT") FORCE_RELATIVE_ERROR = FORCE_RELATIVE_ERROR*0.05
        endif
        if(inode_tot.eq.1) then
            write(22,*) "FORCE_RELATIVE_ERROR     = ", FORCE_RELATIVE_ERROR
        endif

        ISBF = 1  ! Weile Jia, 2015-1-13
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        i = 0
        niter0 = 0
        niter0_ = 0
        NUM_SCF0 = 0
        do while(.true.)
            i = i + 1
            temp_char = 'SCF_ITER0_'//char(i+48)
            temp_char = adjustl(temp_char)
            call read_key_words (9, trim(temp_char), len(trim(temp_char)), right, ierr)
            if (ierr == 0) then
                read (right, *) niter0_
                niter0 = niter0 + niter0_
                NUM_SCF0 = NUM_SCF0 + 1
            else
                exit
            end if
        end do

        if (niter0 > mscf_1) then
            message="niter0 > MAX_SCF_STEP"
            call error_stop(message,__FILE__,__LINE__)
        end if
        !-->have got the niter0
        if (niter0 > 0) then
            mCGbad0=5
        else
            if( CONVERGE .eq. "EASY") then
                niter0 = 100
                !            niter0 = 30
            else if(CONVERGE .eq. "DIFFICULT") then
                niter0 = 100
            else
                !            write(*,*) "CONVERGE should be set as 'DIFFICULT/EASY' "
                niter0 = 100
            end if
            nline0 = 4
            mCGbad0 = 5
            if ( trim ( JOB ) .eq. "NONSCF" ) then
                niter0=50
                if(is_fsm) then
                    nline0=500
                endif
            endif
            !if ( trim(JOB).eq."RELAX" ) then
            !    if (use_hse == 1) then
            !        niter0=50
            !    else
            !        niter0=20
            !    endif
            !endif
        end if
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        call read_key_words(9,'ICGMTH',LEN('ICGMTH'),right,ierr);
        if(ierr .eq. 0) then
            read(right,*) icgmth;
        else
            icgmth=3;
        endif
        call read_key_words(9,'ISCFMTH',LEN('ISCFMTH'),right,ierr);
        if(ierr .eq. 0) then
            read(right,*) iscfmth;
        else
            iscfmth=1;
        endif
        call read_key_words(9,'FERMIDE',LEN('FERMIDE'),right,ierr);
        if(ierr .eq. 0) then
            read(right,*) fermidE;
            fermide=fermide/hartree_ev
        else
            call read_key_words (9, "SYS_TYPE", len("SYS_TYPE"), right, ierr)
            if (ierr /= 0) then
                sys_type = 1  ! semi-conductor, insulator
                fermidE=0.025d0/Hartree_ev
            else
                read (right, *) sys_type
                select case (sys_type)
                case (1) ! semi-conductor, insulator
                    fermidE=0.025d0/Hartree_ev
                case (2) ! metal
                    fermidE=0.200d0/Hartree_ev
                case default
                    sys_type = 1  ! semi-conductor, insulator
                    fermidE=0.025d0/Hartree_ev
                end select 
            end if
            if (inode_tot.eq.1) then
                write (22, *) "SYS_TYPE   = ", sys_type
            end if
            !fermidE=0.025d0/Hartree_ev;
        end if
        call read_key_words(9,'ITYPEFERMI',LEN('ITYPEFERMI'),right,ierr);
        if(ierr .eq. 0) then
            read(right,*)  itypeFermi;
        else
            itypeFermi=2;
        endif


        in_iproj3_CC_2spin=0
        CALL read_key_words ( 9, 'IN.iproj3_cc_2spin', LEN('IN.irpoj3_occ_2spin'), right, IERR )
        if(ierr.eq.0) then 
         read (right, *, IOSTAT=ierror ) right_logical
         if(right_logical) in_iproj3_CC_2spin=1
        endif

        iproj_occ=-1    ! use this to indicate there is no IN.OCC, strange
        !  (should be clear up later, about iproj_occ and itypeFermi)
        iproj22_occ=0
        iproj33_occ=0
        iproj11_occ=0
        CALL read_key_words ( 9, 'IN.OCC', LEN('IN.OCC'), right, IERR )
        if(ierr.eq.0) then 
            read (right, *, IOSTAT=ierror ) right_logical,iproj_occ
            if(right_logical) then
            if(iproj_occ.eq.22) then
              iproj_occ=2
              iproj22_occ=1
             endif
            if(iproj_occ.eq.33) then
              iproj_occ=3
              iproj33_occ=1
             endif
            if(iproj_occ.eq.11) then
              iproj_occ=1
              iproj11_occ=1
             endif
             endif

            if(ierror.ne.0) then
                read(right,*) right_logical
                if(right_logical) then
                iproj_occ=0
                endif
            endif
       ! iproj_occ=0:  use IN.OCC and E_i sequence  to occupied phi_i
       ! iproj_occ=1:  use IN.OCC and <psi_in_j|phi_i>, to identify phi_I and occupy phi_i
       ! iproj_occ=2:  use IN.OCC and <psi_in_j|phi_i>, and special algorithm to
       ! construct the occupation for exceptional j, on top of Fermi-Dirac
       ! iporj_occ=3"  use o_j in IN.OCC and psi'_j= \sum_j <phi_i|psi_in_j> phi_i to occupy phi'_j 
       ! This is used to constraint the occupation of some special state, and perhaps prepare for TDDFT 



            if(inode_tot.eq.1) then
                write(22,*) "IN.OCC   = ", right_logical,iproj_occ
            endif
        else
            if(inode_tot.eq.1) then
                write(22,*) "IN.OCC    = ", .FALSE.,iproj_occ
            endif
        endif

        ! iproj_occ.eq.-1  : no IN.OCC, use Fermi
        ! iproj_occ.eq.0   : simple IN.OCC, not use Fermi
        ! iproj_occ.eq.1   : projection, IN.OCC, not use Fermi
        ! iproj_occ.eq.2   : phase projection, IN.OCC as additional, but use Fermi
        ! iproj_occ.eq.3   : phase projection, use IN.OCC, not use Fermi

            if(iproj_occ.eq.-1.or.iproj_occ.eq.2) then
            itypeFermi=1      ! normal one, without IN.OCC, might be replaced by explicit input
            elseif(iproj_occ.eq.0.or.iproj_occ.eq.1.or.iproj_occ.eq.3) then
            itypeFermi=0      ! directly use the IN.OCC value
            endif

        if(iproj_occ.eq.1.and.iwg_in.eq.0) then
            write(6,*) "for IN.OCC and iproj_occ=1, must have IN.WG"
            write(22,*) "for IN.OCC and iproj_occ=1, must have IN.WG"
            call mpi_abort(mpi_comm_world,ierr)
        endif
        if(iproj_occ.eq.2.and.iwg_in.eq.0) then
            write(6,*) "for IN.OCC and iproj_occ=2, must have IN.WG"
            write(22,*) "for IN.OCC and iproj_occ=2, must have IN.WG"
            call mpi_abort(mpi_comm_world,ierr)
        endif
        if(iproj_occ.eq.3.and.iwg_in.eq.0) then
            write(6,*) "for IN.OCC and iproj_occ=2, must have IN.WG"
            write(22,*) "for IN.OCC and iproj_occ=2, must have IN.WG"
            call mpi_abort(mpi_comm_world,ierr)
        endif

        if(iproj33_occ.eq.1.and.in_iproj3_CC_2spin.eq.0) then
       write(6,*) "for IN.OCC and iproj_occ=33, must have IN.iproj3_CC_2spin"
       write(22,*) "for IN.OCC and iproj_occ=33, must have IN.iproj3_CC_2spin"
            call mpi_abort(mpi_comm_world,ierr)
        endif


        if(iproj_occ.eq.3) then
        CALL read_key_words ( 9, 'PROJ3_DETAIL', LEN('PROJ3_DETIAL'), right, IERR )
        if(ierr.eq.0) then
        read(right,*) Ecut_occ_iproj3, dEcut_occ_iproj3
        Ecut_occ_iproj3=Ecut_occ_iproj3/27.211396d0
        dEcut_occ_iproj3=dEcut_occ_iproj3/27.211396d0
        else
        Ecut_occ_iproj3=0.d0
        dEcut_occ_iproj3=-1.d0
        endif
        endif



        CALL read_key_words ( 9, 'IN.CC', LEN('IN.CC'), right, IERR )
        if(ierr.eq.0) then 
            read ( right, * ) right_logical
            if( right_logical ) then
                itypeFermi=-1
                if(inode_tot.eq.1) then
                    write(22,*) "IN.CC   = ", right_logical
                endif
            else
                if(inode_tot.eq.1) then
                    write(22,*) "IN.CC    = ", .FALSE.
                endif
            endif
        else
            if(inode_tot.eq.1) then
                write(22,*) "IN.CC    = ", .FALSE.
            endif
        endif

        TDDOS_OCC=.false.
        CALL read_key_words ( 9, 'IN.OCC_ADIA', LEN('IN.OCC_ADIA'), right, IERR )
        if(ierr.eq.0) then 
            read ( right, * ) right_logical
            if( right_logical ) then
                TDDOS_OCC=.true.
                if(inode_tot.eq.1) then
                    write(22,*) "IN.OCC_ADIA   = ", right_logical
                endif
            else
                if(inode_tot.eq.1) then
                    write(22,*) "IN.OCC_ADIA    = ", .FALSE.
                endif
            endif
        else
            if(inode_tot.eq.1) then
                write(22,*) "IN.OCC_ADIA    = ", .FALSE.
            endif
        endif



        !--> set the default
        do i=1,niter0
            icgmth0(i)=icgmth
            fermide0(i)=fermide
            iscfmth0(i)=iscfmth
            amx_mth0(i)=0.d0
            itypeFermi0(i)=itypeFermi
        enddo
        iscfmth0(1:6)=0
        if(trim(JOB).eq."NONSCF") then
            do i=1,niter0
                iscfmth0(i)=0
            enddo
        endif
        !
        i = 0
        do j = 1, NUM_SCF0
            temp_char = 'SCF_ITER0_'//char(j+48)
            temp_char = adjustl(temp_char)
            call read_key_words (9, trim(temp_char), len(trim(temp_char)), right, ierr)
            if (ierr == 0) then
                read (right, *) niter0_, nline0, iCGmth0_, aiscfmth_, FermidE0_, itypeFermi0_
                if (inode_tot.eq.1) then
                    write (right, '(a,x,i3,x,i3,x,i4,2x,f8.4,2x,f10.5,2x,i3)') ' '//trim(temp_char)//' = ', niter0_, nline0, iCGmth0_, aiscfmth_, FermidE0_, itypeFermi0_
                    write (22, '(a)') trim(right)
                end if
                iCGmth0(i+1:i+niter0_) = iCGmth0_
                aiscfmth = aiscfmth_
                FermidE0(i+1:i+niter0_) = FermidE0_
                if(iproj_occ.eq.-1.or.iproj_occ.eq.2) then
                itypeFermi0(i+1:i+niter0_) = itypeFermi0_
                endif
                !
                iscfmth0(i+1:i+niter0_)=aiscfmth+1.D-6
                amx_mth0(i+1:i+niter0_)=aiscfmth-iscfmth0(i+1:i+niter0_)
                FermidE0(i+1:i+niter0_)=FermidE0(i+1:i+niter0_)/Hartree_ev
                !
                i = i + niter0_
                !
            end if
        end do

        if(trim(JOB).eq."DOS") iCGMTH0(1)=-1    ! special for DOS
        if(trim(JOB).eq."MOMENT") iCGMTH0(1)=-2  ! special, used in eigen_solver.f
        if(inode_tot.eq.1) then
            if (NUM_SCF0 < 1) then
                write (right, '(a,x,i3,x,i3,x,i4,2x,f8.4,2x,f10.5,2x,i3)') ' SCF_ITER0_1 = ', 6, nline0, iCGMTH0(1),iscfmth0(1)+amx_mth0(1), FermidE0(1)* Hartree_ev, itypeFermi0(1)
                write (22, '(a)') trim(right)
                write (right, '(a,x,i3,x,i3,x,i4,2x,f8.4,2x,f10.5,2x,i3)') ' SCF_ITER0_2 = ', niter0-6 , nline0, iCGMTH0(7),iscfmth0(7)+amx_mth0(7), FermidE0(7)* Hartree_ev, itypeFermi0(7)
                write (22, '(a)') trim(right)
            end if
        end if
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        i = 0
        niter1 = 0
        niter1_ = 0
        NUM_SCF1 = 0
        do while(.true.)
            i = i + 1
            temp_char = 'SCF_ITER1_'//char(i+48)
            temp_char = adjustl(temp_char)
            call read_key_words (9, trim(temp_char), len(trim(temp_char)), right, ierr)
            if (ierr == 0) then
                read (right, *) niter1_
                niter1 = niter1 + niter1_
                NUM_SCF1 = NUM_SCF1 + 1
            else
                exit
            end if
        end do
        if (niter1 > mscf_1) then
            message="niter1 > MAX_SCF_STEP"
            call error_stop(message,__FILE__,__LINE__)
        end if
        if (niter1 > 0) then
            mCGbad0=5
        else
            ISCF_ITER1=0
            if(CONVERGE .eq. "EASY") then
                niter1 = 40
            else
                niter1 = 50
            endif
            nline1 = 4
            mCGbad1 = 5
            !if ( trim(JOB).eq."RELAX" ) then
            !    if (use_hse == 1) then
            !        niter1=50
            !    else
            !        niter1=10
            !    endif
            !endif
        end if
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !--> set the default
        do i=1,niter1
            iCGMTH1(i)=icgmth
            FermidE1(i)=fermidE
            iscfmth1(i)=iscfmth
            amx_mth1(i)=0.d0
            itypeFermi1(i)=itypeFermi
        end do
        !
        i = 0
        do j = 1, NUM_SCF1
            temp_char = 'SCF_ITER1_'//char(j+48)
            temp_char = adjustl(temp_char)
            call read_key_words (9, trim(temp_char), len(trim(temp_char)), right, ierr)
            if (ierr == 0) then
                read (right, *) niter1_, nline1, iCGmth1_, aiscfmth_, FermidE1_, itypeFermi1_
                if (inode_tot.eq.1) then
                    write (right, '(a,x,i3,x,i3,x,i4,2x,f8.4,2x,f10.5,2x,i3)') ' '//trim(temp_char)//' = ', niter1_, nline1, iCGmth1_, aiscfmth_, FermidE1_, itypeFermi1_
                    write (22, '(a)') trim(right)
                end if
                iCGmth1(i+1:i+niter1_) = iCGmth1_
                aiscfmth = aiscfmth_
                FermidE1(i+1:i+niter1_) = FermidE1_
                if(iproj_occ.eq.-1.or.iproj_occ.eq.2) then
                itypeFermi1(i+1:i+niter1_) = itypeFermi1_
                endif
                !
                iscfmth1(i+1:i+niter1_)=aiscfmth+1.D-6
                amx_mth1(i+1:i+niter1_)=aiscfmth-iscfmth1(i+1:i+niter1_)
                FermidE1(i+1:i+niter1_)=FermidE1(i+1:i+niter1_)/Hartree_ev
                !
                i = i + niter1_
                !
            end if
        end do

        if(inode_tot.eq.1) then
            if (NUM_SCF1 < 1) then
                write (right, '(a,x,i3,x,i3,x,i4,2x,f8.4,2x,f10.5,2x,i3)') ' SCF_ITER1_1 = ', niter1, nline1, iCGMTH1(1),iscfmth1(1)+amx_mth1(1), FermidE1(1)* Hartree_ev, itypeFermi1(1)
                write (22, '(a)') trim(right)
            end if
        end if
        !??????????????????????????????????????????????????????
        !          if(use_hse .ne. 1) then !! not sure about hse 
        !              iscfmth1(1:6)=0  !! for relaxtion or md, do non-scf first 6 steps
        !          endif
        !???????????????????????????????????????????????????????
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words(9,'NONLOCAL',LEN('NONLOCAL'),right,IERR )
        if(ierr.eq.0) then
            READ ( right, * ) ilocal
        else
            ilocal = 2
        endif

        if(inode_tot.eq.1) then
            write(22,1120) ilocal
        endif
        1120    format(" NONLOCAL  = ",i2) 
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        max_rcut=0.d0
        is_rcut_type_set=0
        is_rcut_set=0
        do i=1,ntype
            !temp_char = "IN.PSP_RCUT"//char(48+i)
            write(temp_char,*) i
            temp_char="IN.PSP_RCUT"//ADJUSTL(trim(temp_char))
            temp_char = ADJUSTL(temp_char)
            CALL read_key_words ( 9, TRIM(temp_char),len_trim(temp_char), right, IERR )
            if(ierr.eq.0) then
                READ ( right, * ) rcut_of_type(i)
                is_rcut_type_set=1
            else
                rcut_of_type(i)=upfpsp(i)%r_cut
            endif

            if(rcut_of_type(i)>max_rcut) then
                max_rcut=rcut_of_type(i)
            endif

            !! just test
            !!           write(*,*) "type,rcut=",i,rcut_of_type(i)
        enddo

        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'IN.A_FIELD', LEN('IN.A_FIELD'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, *) is_in_A_field
            if(is_in_A_field) then
                READ ( right, *,IOSTAT=ierror ) is_in_A_field,A_field(1),A_field(2),A_field(3)
            endif
        else
            is_in_A_field=.false.
        endif
        if(is_in_A_field.eq..false.) then
            A_field(1)=0.d0
            A_field(2)=0.d0
            A_field(3)=0.d0
        endif
        write(22,'(1x,A,L,3(F,1x))') 'IN.A_FIELD = ', is_in_A_field,A_field(1:3)
        !a_field_list(3,20), a_field_time_list(20)
        !IN.A_FIELD_LIST1=afield_x,afield_y,afield_z,IN.TDDFT_TIME_1
        !IN.A_FIELD_LIST2=afield_x,afield_y,afield_z,IN.TDDFT_TIME_2
        !IN.A_FIELD_LIST3=afield_x,afield_y,afield_z,IN.TDDFT_TIME_3
        td_a_field_list_count=0
        td_a_field_list=0.d0
        DO i = 1, 100
            temp_char="IN.A_FIELD_LIST"//itostr(i)
            temp_char = ADJUSTL(temp_char)
            CALL read_key_words ( 9, TRIM(temp_char),len_trim(temp_char), right, IERR )
            IF( IERR == 0 ) THEN
                td_a_field_list_count = td_a_field_list_count + 1
                READ ( right, *,IOSTAT=ierror ) td_a_field_list(1,i),td_a_field_list(2,i),td_a_field_list(3,i), td_a_field_list_filename(i)
                temp_char='IN.A_FIELD_LIST'//itostr(i)
                write(22,'(1x,A,3(F,1x),A)') ,ADJUSTL(trim(temp_char))//' = ', td_a_field_list(1:3,i),ADJUSTL(trim(td_a_field_list_filename(i)))
            ELSE
                EXIT
            ENDIF
            IF(i .gt. td_efield_list_max) then
                message="Too many A_FIELD_LIST. The upper limit is "//ADJUSTL(trim(itostr(td_efield_list_max)))
                call error_stop(message,__FILE__,__LINE__)
            ENDIF
        ENDDO


        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        CALL read_key_words ( 9, 'rcut', LEN('rcut'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) rcut
            is_rcut_set=1
        endif

        if(is_rcut_type_set .eq. 1) then
            rcut=max_rcut
        else if(is_rcut_set .eq.1) then
            rcut_of_type(:)=rcut
        else
            rcut_of_type(1:ntype)=upfpsp(1:ntype)%r_cut
            if(ACCURACY .eq. "VERYHIGH") then 
                rcut_of_type(1:ntype)=rcut_of_type(1:ntype)*1.1d0
            endif 
            rcut=maxval(rcut_of_type(1:ntype))
        endif

        if(inode_tot.eq.1) then
            write(22,*) "RCUT      = ", rcut
            do i=1,ntype
                !write(22,*) "IN.PSP_RCUT"//char(48+i)//"   = ",rcut_of_type(i)
                write(temp_char,*) i
                temp_char="IN.PSP_RCUT"//ADJUSTL(trim(temp_char))
                write(22,*) ADJUSTL(trim(temp_char)),' =',rcut_of_type(i)
            enddo
        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if(trim(JOB).eq."MD".or.trim(JOB).eq."TDDFT".or.trim(JOB).eq."NAMD") then
            CALL read_key_words(9,'MD_DETAIL',LEN('MD_DETAIL'),right,IERR)
            if(ierr.eq.0) then
                READ ( right, * ) iMD,MDstep,dtMD,Temperature1,Temperature2
                if(inode_tot.eq.1) then
                    write(22,2006) iMD,MDstep,dtMD,Temperature1,Temperature2
                endif
                iMD=iMD
                istddft=.false.
                if(trim(JOB).eq."TDDFT") then
                    istddft=.true.
                    dt_step=dtMD


                    itddft_nob=0

                    CALL read_key_words(9,'TDDFT_DETAIL',LEN('TDDFT_DETAIL'),right,IERR)
                    if(ierr.eq.0) then
                        READ ( right, * ) m1_td,m2_td,mstate_td
                        if(mstate_td.gt.0) then
                        if(.not.(m1_td.ge.1 .and. m1_td .le. mx .and. m2_td.ge.m1_td .and. m2_td .le. mx .and. mstate_td .ge. m1_td .and. mstate_td .le. m2_td)) then
                            if(inode_tot.eq.1) then
                                write(*,*) "wrong TDDFT_DETAIL. m1 in [1,NUM_BAND],m2 in [m1,NUM_BAND],mstate in [m1,m2]"
                                call mpi_abort(mpi_comm_world,ierr)
                            endif
                        endif
                        else
                        if(.not.(m1_td.ge.1 .and. m1_td .le. mx .and. m2_td.ge.m1_td .and. m2_td .le. mx)) then
                            if(inode_tot.eq.1) then
                                write(*,*) "wrong TDDFT_DETAIL. m1 in [1,NUM_BAND],m2 in [m1,NUM_BAND],mstate in [m1,m2]"
                                call mpi_abort(mpi_comm_world,ierr)
                            endif
                        endif
                        endif

                    else
                        m1_td=1
                        m2_td=mx
                        mstate_td=mx
                    endif

                    mstate_td_tmp=mstate_td

                    if(mstate_td.gt.0) then
                    mst_win0=m1_td
                    mst_win=m2_td-m1_td+1
                    mst_td=mstate_td-m1_td+1
                    else    ! if mst_td.lt.0, then itddft_nob 
                    itddft_nob=1
                    mst_win0=m1_td
                    mst_win=m2_td-m1_td+1
                    mst_td=mst_win   ! for itddft_nob, mst_td=mst_win
                    endif

                    tolInt_td=1.d-4
                    minInt_td=100
                    maxInt_td=1000
                    if(inode_tot.eq.1) then
                        !write(22,2106) mst_win0,mst_win,mst_td
                        write(22,2106) m1_td,m2_td,mstate_td_tmp 
                        2106   format(" TDDFT_DETAIL = ",3(i6,2x))
                    endif

                    if(itddft_nob.eq.1) then
                     allocate(numb_td(nkpt,islda))
                     allocate(list_select(mst_win,nkpt,islda))
                     allocate(select_weight(mst_win,nkpt,islda))
                     do iislda=1,islda
                     do kpt=1,nkpt
                     numb_td(kpt,iislda)=totNel_spin(iislda)+0.1-(mst_win0-1)
                     enddo
                     enddo
                     if(is_so.eq.1.or.is_som.eq.1) then
                     do kpt=1,nkpt
                     numb_td(kpt,1)=num_electron+0.1-(mst_win0-1)
                     enddo
                     endif
                    endif


                    iscale_kin_boltzmann=0

                    if(itddft_nob.eq.1) then
                    CALL read_key_words(9,'TDDFT_NOB',LEN('TDDFT__NOB'),right,IERR)
                    if(ierr.eq.0) then
                      READ ( right, * )   &
                 irandom_tddft,S_td_c,tau_dephase,temp,iscale_kin_boltzmann,iselect_nob_opt
                    else
                        write(6,*) "for TDDFT_NOB, must have TDDFT_NOB input"
                        stop
                    endif

                          beta_kt_TDDFT=temp*0.025/300/27.211396
                          allocate(tau_dephase_m(mx))
                          if(tau_dephase.ge.0.d0) then
                                tau_dephase_m=tau_dephase
                          else 
                                open(14,file="IN.BOLTZMANN_TAU", iostat=ierr) 
                                rewind(14)
                                read(14,*) (tau_dephase_m(m1),m1=1,mst_win)
                                close(14)
                          endif
                     endif


                    !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                    itddft_boltzmann=0
                    CALL read_key_words(9,'TDDFT_BOLTZMANN',LEN('TDDFT_BOLTZMANN'),right,IERR)
                    if(ierr.eq.0) then
                      READ ( right, *, IOSTAT=ierror) itddft_boltzmann, iscale_kin_boltzmann,temp,tau_dephase,istep_boltzmann_start,nstep_linear_boltzmann
                      if(ierror.ne.0) then
                        nstep_linear_boltzmann=1000
                      READ ( right, *, IOSTAT=ierror) itddft_boltzmann, iscale_kin_boltzmann,temp,tau_dephase,istep_boltzmann_start
                      if(ierror.ne.0) then
                        READ ( right, *, IOSTAT=ierror) itddft_boltzmann, iscale_kin_boltzmann,temp,tau_dephase
                          if(ierror.ne.0) then
                            write(6,*) "MESSAGE: in etot.input, the TDDFT_BOLTZMANN LINE IS NOT CORRECT"
                            stop
                          endif
                        istep_boltzmann_start=1
                      endif

                      endif


                        if(itddft_boltzmann.eq.1) then
                            allocate(DD_integral(mx,mx))
                            allocate(DD_integral0(mx,mx))
                            allocate(tau_dephase_m(mx))
                            allocate(dc_linear_boltzmann(mx,mst_td,nkpt,islda))
                            DD_integral=0.d0
                            DD_integral0=0.d0
                            !TCD_All=dcmplx(0.d0,0.d0)
                            beta_kt_TDDFT=temp*0.025/300/27.211396
                            dc_linear_boltzmann=cmplx(0.d0,0.d0)
                            if(tau_dephase.ge.0.d0) then
                                tau_dephase_m=tau_dephase
                            else 
                                open(14,file="IN.BOLTZMANN_TAU", iostat=ierr) 
                                rewind(14)
                                read(14,*) (tau_dephase_m(m1),m1=1,mst_win)
                                close(14)
                            endif
                        endif


                    else
                        itddft_boltzmann=0
                    endif
    !ccccccccccccccccccccccccccccccccccccccccccccccccccccc

                      if(itddft_boltzmann.eq.1.or.itddft_nob.eq.1) then
                      allocate(TCD_ALL(mx,mx,nkpt,islda))
                      allocate(TCD_alambda(mst_win,mst_win,nkpt,islda))
                      TCD_all=cmplx(0.d0,0.d0)
                      TCD_alambda=cmplx(0.d0,0.d0)
                      beta_kt_TDDFT=temp*0.025/300/27.211396
                      endif

                      if(iscale_kin_boltzmann.eq.3) then
                      allocate(fatom_TCD(3,matom_1))
                      endif

             !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


                    CALL read_key_words(9,'OUT.TDDFT',LEN('OUT.TDDFT'),right,IERR)
                    if(ierr.eq.0) then
                        READ ( right, * ) isTddftOut1,isTddftOut1_Cmat,dtTddftOut1,isTddftout2,dtTDDFTOut2 
                    else
                        isTddftOut1=.false.
                        dtTddftOut1=1.0
                        isTddftOut1_Cmat=.false.
                        isTddftOut2=.false.
                        dtTddftOut2=1.0
                    endif
                    if(inode_tot.eq.1) then
                        write(22,2107) isTddftOut1,isTddftOut1_Cmat,dtTddftOut1,isTddftOut2,dtTddftOut2
                        2107 format(" OUT.TDDFT =",L4,1x,L4,1x,E13.5,2x,L4,1x,E13.5)
                    endif
                endif  ! TDDFT

                if(trim(JOB).eq."TDDFT".or.trim(JOB).eq."NAMD") then
                    CALL read_key_words(9,'TDDFT_SPACE',LEN('TDDFT_SPACE'),right,IERR)
                    a_tddft(:)=0.d0
                    if(ierr.eq.0) then
                        READ ( right, *) itddft_space
                        if(itddft_space.ge.2 .and. itddft_space.le.4) then
                            READ ( right, *) itddft_space,num_par,(a_tddft(i),i=1,num_par)
                        endif
                        if(itddft_space .eq. 99) then
#ifndef PYTHON
                            if(inode_tot .eq.1) then
                                write(*,*) "ispace_type .eq. 99 not supported."
                                call mpi_abort(mpi_comm_world,ierr)
                            endif
#endif
                        endif
                    else
                        itddft_space=0
                    endif
                    CALL read_key_words(9,'TDDFT_TIME',LEN('TDDFT_TIME'),right,IERR)
                    b_tddft(:)=0.d0
                    if(ierr.eq.0) then
                        READ ( right, *) itddft_time
                        if(itddft_time.eq.2.or.itddft_time.eq.22) then
                            READ ( right, *) itddft_time,num_par,(b_tddft(i),i=1,num_par)
                        endif
                        if(itddft_time .eq. 99) then
#ifndef PYTHON
                            if(inode_tot .eq.1) then
                                write(*,*) "itime_type .eq. 99 not supported."
                                call mpi_abort(mpi_comm_world,ierr)
                            endif
#endif
                        endif
                    else
                        itddft_time=0
                    endif
                    CALL read_key_words(9,'TDDFT_STIME',LEN('TDDFT_STIME'),right,IERR)
                    if(ierr.eq.0) then
                        READ ( right, * ) tddft_stime
                    else
                        tddft_stime=0.d0
                        !MD,TDDFT,NAMD use tddft_stime as the start time
                    endif
                    ftddft_time_svalue=0.d0
                    !write(22,2108) tddft_stime
                    !2108 format(" TDDFT_STIME =",1x,E13.5)
                    nstep_prev=0.2+tddft_stime/dtMD-1 ! used for namd.
                    ! tddft_stime/dtMD position for xatom
                    ! tddft_stime/dtMD-1 position for wg
                    ! nstep_prev is used to output wg
                endif  ! TDDFT
                if(trim(JOB).eq."MD") then
                    CALL read_key_words(9,'MD_STIME',LEN('MD_STIME'),right,IERR)
                    if(ierr.eq.0) then
                        READ ( right, * ) tddft_stime
                    else
                        tddft_stime=0.d0
                    endif
                    !write(22,2109) tddft_stime
                    !nstep_prev=0.2+tddft_stime/dtMD
                    !2109 format(" MD_STIME =",1x,E13.5)
                endif

                !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 9, 'IN.OCC_T', LEN('IN.OCC_T'), right, IERR )
                if(ierr.eq.0) then 
                    read ( right, * ) right_logical
                    IF ( right_logical ) THEN
                        iocc_in_time_TDDFT  = 1
                        if(inode_tot.eq.1) then
                            write(22,*) "IN.OCC_T    = ", right_logical
                        endif
                        open(10,file="IN.OCC_T")
                        rewind(10)
                        read(10,*) num_kpt,num_occ_t_TDDFT
                        allocate(ind_occ_t(num_occ_t_TDDFT,num_kpt,islda))
                        allocate(iform_occ(num_occ_t_TDDFT,num_kpt,islda))
                        allocate(data_occ_t(5,num_occ_t_TDDFT,num_kpt,islda))
                        do ikpt=1,num_kpt
                            read(10,*) 
                            do iline=1,num_occ_t_TDDFT
                                read(10,*)ind_occ_t(iline,ikpt,1),iform_occ(iline,ikpt,1),(data_occ_t(id,iline,ikpt,1),id=1,5)
                            enddo
                        enddo
                        close(10)
                        if(islda.eq.2) then
                            open(10,file="IN.OCC_T_2")
                            rewind(10)
                            read(10,*) num_kpt,num_occ
                            do ikpt=1,num_kpt
                                read(10,*) 
                                do iline=1,num_occ_t_TDDFT
                                    read(10,*)ind_occ_t(iline,ikpt,2),iform_occ(iline,ikpt,2),(data_occ_t(id,iline,ikpt,2),id=1,5)
                                enddo
                            enddo
                            close(10)
                        endif
                    ELSE
                        iocc_in_time_TDDFT  = 0
                        if(inode_tot.eq.1) then
                            write(22,*) "IN.OCC_T   = ", right_logical
                        endif
                    ENDIF
                else
                    iocc_in_time_TDDFT  = 0
                    if(inode_tot.eq.1) then
                        write(22,*) "IN.OCC_T    = ", .FALSE.
                    endif
                endif



                !cccccccccccccccccccccccccccccccccccccccccccccccccccccc

                isnamd = .false.
                icycle_namd=0
                if(trim(JOB).eq."NAMD") then
                    isnamd=.true.
                    CALL read_key_words(9,'NAMD_DETAIL',LEN('NAMD_DETAIL'),right,IERR)
                    if(ierr.eq.0) then
                        READ (right,*,IOSTAT=ierror)  &
                          m1_td,m2_td,mstep_wave_out,icycle_namd,nstep_cycle,iflag_namd_crossk
                        if(ierror.ne.0) then
                        icycle_namd=0
                        nstep_cycle=1
                        iflag_namd_crossk=0
                        READ ( right, * ) m1_td,m2_td,mstep_wave_out
                        endif

                        mst_win0=m1_td
                        mst_win=m2_td-m1_td+1
                        if(mstep_wave_out.ne.0) then
                            write(22,*) "#WARNING, mstep_wave_out can produce a large wave function output file ugio_allxxx"
                            write(22,*) "#WARNING, remember to remove ugio_allxxx"

                        endif
                        if(MDstep/mstep_wave_out.gt.50) then
                            mstep_wave_out=MDstep/50
                            write(22,*) "#WARNING, total output ug.gt.50"
                            write(22,*) "#WARNING: change mstep_wave_out to", mstep_wave_out
                        endif

                        if(mstep_wave_out.lt.0) mstep_wave_out=-mstep_wave_out

                    else
                        mst_win0=1
                        mst_win=mx
                        mstep_wave_out=0
                    endif
                endif

                if(trim(JOB).eq."NAMD") then
                    CALL read_key_words(9,'NAMD_SPECIAL',LEN('NAMD_SPECIAL'),right,IERR)
                    if(ierr.eq.0) then
                        READ ( right, * )iflag_NAMD_sp,m1_namd_sp,m2_namd_sp
                    else
                        iflag_NAMD_sp=0
                    endif
                endif


                !JOB="MD"
                !JOB_1=trim(JOB)
            else
                iMD = 0
                MDstep=0
                message="For JOB .eq. MD .or. TDDFT, must have MD_DETAIL with MDmethod, MDstep, dtMD, Temperature1, Temperature2"
                call error_stop(message,__FILE__,__LINE__)
            endif
        else
            iMD = 0
            MDstep=0
        endif
        2006   format(" MD_DETAIL =",i4,2x,i6,2x,f10.5,2x,f10.5,1x,f10.5)


        if(trim(JOB).eq."MD") then
            iflag_MD_sp=0
            CALL read_key_words(9,'MD_SPECIAL',LEN('MD_SPECIAL'),right,IERR)
            if(ierr.eq.0) then
              read(right, *) iflag_MD_sp 
              if(iflag_MD_sp.eq.3.or.iflag_MD_sp.eq.33) then
              read(right, *) iflag_MD_sp,x10_MD_sp,x20_MD_sp,x30_MD_sp,Rcut_MD_sp,dR_MD_sp,V_MD_sp, &
                    frac_MD_sp,P_MD_sp,R_rate_sp
              P_MD_sp=P_MD_sp/2.9417D+8 ! convert bar(0.9869 atom) to Hartree/Bohr^3 
              else
              read(right, *) iflag_MD_sp,x10_MD_sp,x20_MD_sp,x30_MD_sp,Rcut_MD_sp,dR_MD_sp,V_MD_sp 
              endif
             Rcut_MD_sp=Rcut_MD_sp/0.529177d0
             dR_MD_sp=dR_MD_sp/0.529177d0
             V_MD_sp=V_MD_sp/27.211396
            else
              iflag_MD_sp=0
            endif
         endif

        if(trim(JOB).eq."MD") then
            iflag_MD_sp_solvent=0
            CALL read_key_words(9,'MD_SPECIAL2',LEN('MD_SPECIAL2'),right,IERR)
            if(ierr.eq.0) then
              read(right, *) iflag_MD_sp_solvent,dRcut_MD_sp 
              dRcut_MD_sp=dRcut_MD_sp/0.529177d0
            else
              iflag_MD_sp_solvent=0
            endif
         endif

              

        if(JOB=="MD") then
            CALL read_key_words ( 9, 'IN.MDOPT', LEN('IN.MDOPT'), right, IERR )
            if(iMD==8 .or. iMD==88) then 
                MCTRL_is_MSST=.true.
            else
                MCTRL_is_MSST=.false.
            endif
            if(ierr.eq.0) then
                READ ( right, * ) right_logical 
                if(right_logical) then
                    call read_mdopt(dtMD,Temperature1,temperature2,imov_at,natom,AL)
                else
                    call default_mdopt(dtMD,Temperature1,temperature2,imov_at,natom,AL)
                endif
                if(inode_tot.eq.1) then
                    write(22,*) "IN.MDOPT    = ", right_logical
                endif
            else
                call default_mdopt(dtMD,Temperature1,temperature2,imov_at,natom,AL)
                if(inode_tot.eq.1) then
                    write(22,*) "IN.MDOPT    = ", .false.
                endif
            endif
            if(iMD==4 .or. iMD==44) MCTRL_stress=1
            if(iMD==5 .or. iMD==55) MCTRL_stress=1
            if(iMD==7 .or. iMD==77) MCTRL_stress=1
            if(iMD==8 .or. iMD==88) MCTRL_stress=1
            call mpi_barrier(mpi_comm_world, ierr)
        endif
        if(JOB=="TDDFT") then
            CALL read_key_words ( 9, 'IN.TDDFTOPT', LEN('IN.TDDFTOPT'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, * ) right_logical 
                if(right_logical) then
                    call read_tddftopt()
                else
                    call default_tddftopt()
                endif
                if(inode_tot.eq.1) then
                    write(22,*) "IN.TDDFTOPT    = ", right_logical
                endif
            else
                call default_tddftopt()
                if(inode_tot.eq.1) then
                    write(22,*) "IN.TDDFTOPT    = ", .false.
                endif
            endif
            call mpi_barrier(mpi_comm_world, ierr)
        endif

        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if(iwg_in.eq.0.and.JOB.eq."NAMD".and.iMD.eq.11) then
            if(inode_tot.eq.1) then
                write(22,*) "FOR JOB=NAMD,iMD=11, must have IN.WG=T(from previous step), stop"
                call mpi_abort(mpi_comm_world,ierr)
            endif
        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        if(iMD.eq.1.or.iMD.eq.11) then
            CALL read_key_words ( 9, 'MD_VV_SCALE', LEN('MD_VV_SCALE'), right, IERR )
            if(ierr.eq.0) then
                READ ( right, * )  nstep_temp_VVMD
                !ccccc actually, iscale_temp_VVMD is redundant. 
                !ccccc when iscale_temp_VVMD=0, nstep_temp_VVMD is not used, energy is automatically scaled
                !ccccc every 100 steps. This can be replaced by nstep_temp_VVMD=100 for the default. 
                !ccccc when iscale_tmp_VVMD=1, the kinetic energy is scaled every nstep_temp_VVMD
                !cccc iscale_temp_VVMD no longer used
                !cccc               iscale_temp_VVMD = 0            ! whether to scale VVMD
                nstep_temp_VVMD =  nstep_temp_VVMD     ! every nstep to scale
            else
                nstep_temp_VVMD =100      ! default value
            endif
        else
            nstep_temp_VVMD =100
        endif
        if(inode_tot.eq.1) then
            write(22,2007) nstep_temp_VVMD 
            2007          format(" MD_VV_SCALE = ", i6)
        endif
        !!! ldau init
        do i=1,ntype    ! This is reread in from ldau_init
            !temp_char = "LDAU_PSP"//char(48+i)
            write(temp_char,*) i
            temp_char="LDAU_PSP"//ADJUSTL(trim(temp_char))
            temp_char = ADJUSTL(temp_char)
            CALL read_key_words ( 9, TRIM(temp_char),len_trim(temp_char), right, IERR )
            if(ierr.eq.0) then
                is_ldau_type_tmp=1
                READ ( right, *, iostat=ierror ) LDAU_l_tmp, Hubbard_U_tmp(1),Hubbard_U_tmp(2)
                if(ierror.ne.0) then
                READ ( right, *) LDAU_l_tmp, Hubbard_U_tmp(1)
                Hubbard_U_tmp(2)=Hubbard_U_tmp(1)
                endif

                Hubbard_U_tmp=Hubbard_U_tmp/Hartree_ev

                if(LDAU_l_tmp .lt. 0) then
                    is_LDAU_type_tmp=0
                    LDAU_l_tmp=-1
                    Hubbard_U_tmp=1.0e-10
                endif
            else
                Hubbard_U_tmp=1.0e-10
                is_LDAU_type_tmp=0
                LDAU_l_tmp=-1
            endif
            if(inode_tot .eq. 1) then
                write(22,*) TRIM(temp_char)//"   = ",LDAU_l_tmp,Hubbard_U_tmp*Hartree_ev," eV"
            endif
            if(is_ldau_type_tmp.eq.1) then
                is_LDAU=.true.
            endif
        enddo
        !
        if(trim(JOB).eq."DOS") then
            is_LDAU=.false.
        endif
        !
        if(ipsp_all_1000 .eq. 1) then 
            if(use_hse.eq.1 .and. JOB_1 .ne. "NONSCF") then
                if(precision_flag .eq. auto_prec) then
                precision_flag = double_prec
                endif
            endif
        endif
        if(is_LDAU) then
            call ldau_init(ldau_typ,'etot.input',natom,ntype,nkpt,islda,upfpsp,iatom,precision_flag,is_SOM) 
            ! The above parameters are rewread inside the ldau_init
        endif
        !ldau not support uspp
        !ldau+som not support stress
        !ldau+som not support force
        !soc not support stress
        !ldau not support wkm
        !
        if(is_SO .and. ipsp_all==2) then
            write(message,*) "SPIN=22 or SPIN=222 not support ultra-soft pseudopotential, stop"
            call error_stop(message,__FILE__,__LINE__)
        endif
        if(is_LDAU .and. ipsp_all==2) then
            write(message,*) "LDA+U not support ultra-soft pseudopotential, stop"
            call error_stop(message,__FILE__,__LINE__)
        endif
        if(is_SO == 1 .and. istress_cal == 1) then
            write(message,*) "SPIN=22 or SPIN=222 not support stress calculation, stop"
            call error_stop(message,__FILE__,__LINE__)
        endif
        if(is_LDAU .and. is_SOM == 1 .and. iforce == 1) then
            write(message,*) "LDA+U and SPIN=222 not support stress calculation, stop"
            call error_stop(message,__FILE__,__LINE__)
        endif
        if(is_LDAU .and. iflag_wkm == 1) then
            write(message,*) "LDA+U not support WKM, stop"
            call error_stop(message,__FILE__,__LINE__)
        endif
        if((is_LDAU .or. is_SO == 1) .and. iflag_wkm == 1) then
            write(message,*) "LDA+U or SPIN=22 or SPIN=222 not support WKM, stop"
            call error_stop(message,__FILE__,__LINE__)
        endif
        if((is_LDAU .or. is_SO == 1) .and. iflag_fixw == 1) then
            write(message,*) "LDA+U or SPIN=22 or SPIN=222 not support FIXW, stop"
            call error_stop(message,__FILE__,__LINE__)
        endif
        if( is_LDAU .and. is_SO == 1 .and. is_SOM .ne. 1) then
            write(message,*) "LDA+U not support SPIN=22, maybe you want spin=222, stop"
            call error_stop(message,__FILE__,__LINE__)
        endif
        if( is_LDAU .and. ismgga) then
            write(message,*) "LDA+U not support meta-GGA, stop"
            call error_stop(message,__FILE__,__LINE__)
        endif
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'NUM_BLOCKED_PSI', LEN('NUM_BLOCKED_PSI'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) inumber_block_psi
            if(inumber_block_psi .lt. 1) inumber_block_psi=1
        else
            inumber_block_psi= 1     ! Gspace implementation, Real-space: 2
        endif

        if(inode_tot.eq.1) then
            write(22,*) "NUM_BLOCKED_PSI= ", inumber_block_psi
        endif
        if (inumber_block_psi > 1 .and. use_hse == 1) then
            if (inode_tot == 1) write (6, *) "HSE not support NUM_BLOCKED_PSI >= 1" 
            call mpi_abort (mpi_comm_world, ierr)
        end if
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        CALL read_key_words ( 9, 'WF_STORE2DISK', LEN('WF_STORE2DISK'), right, IERR )
        if(ierr .eq. 0) then
            READ ( right, * ) iwrite
            if(iwrite .eq. 1)  storeWf2Memory_7 = 0
            if(iwrite .eq. 0)  storeWf2Memory_7 = 1
            goto 2189
        endif

        CALL read_key_words ( 9, 'WRITE2MEMORY', LEN('WRITE2MEMORY'), right, IERR )
        if(ierr.eq.0) then
            READ ( right, * ) iwrite
        else
            iwrite = 1      ! by default , store the WF to memory.
        endif
        if(iwrite .eq. 1)  then 
            iwrite = 0 
        else 
            iwrite = 1  
        endif
        if(iwrite .eq. 1)  storeWf2Memory_7 = 0
        if(iwrite .eq. 0)  storeWf2Memory_7 = 1
        2189   continue

        ! pay attention to this
        if(PWSCF_OUTPUT .eq. .true.) then
            iwrite=1
            storeWf2Memory_7=0
        endif

        !CALL read_key_words(9,'TDDOS_OCC',LEN('TDDOS_OCC'),right,IERR)
        !if(ierr.eq.0) then
        !    READ ( right, * ) TDDOS_TYE,TDDOS_OCC 
        !else
        !    TDDOS_TYE=0
        !    TDDOS_OCC=.false.
        !endif
        !if(TDDOS_TYE.eq.0) then
        !    TDDOS_OCC=.false.
        !endif
        if(inode_tot.eq.1) then
            write(22,*) "WF_STORE2DISK  = ", iwrite
            !write(22,*) "WF_STORE2MEM = ", storeWf2Memory_7  !for debug.
        endif
        CALL read_key_words ( 9, 'NUM_DOS_GRID', LEN('NUM_DOS_GRID'), right, IERR )
        if (ierr == 0) then
            read (right, *) nE_1
        else
            nE_1 = 4000
        ENDIF
        if(inode_tot.eq.1) then
            write(22,*) "NUM_DOS_GRID = ", nE_1
        endif
        CALL read_key_words ( 9, 'NMAP_MAX', LEN('NMAP_MAX'), right, IERR )
        if (ierr == 0) then
            read (right, *) MCTRL_NMAP_MAX
        else
            MCTRL_NMAP_MAX = 50000
        ENDIF
        if(inode_tot.eq.1) then
            write(22,*) "NMAP_MAX = ", MCTRL_NMAP_MAX
        endif

        CALL read_key_words ( 9, 'KERK_AMIN', LEN('KERK_AMIN'), right, IERR )
        if (ierr == 0) then
            read (right, *) kerk_amin
        else
            kerk_amin=0.3d0
        ENDIF
        if(inode_tot.eq.1) then
            write(22,*) "KERK_AMIN = ", kerk_amin
        endif

        CALL read_key_words ( 9, 'KERK_AMIX', LEN('KERK_AMIX'), right, IERR )
        if (ierr == 0) then
            read (right, *) kerk_amix
        else
            kerk_amix=0.4d0
        ENDIF
        if(inode_tot.eq.1) then
            write(22,*) "KERK_AMIX = ", kerk_amix
        endif

        CALL read_key_words ( 9, 'KERK_AMIX_MAG', LEN('KERK_AMIX_MAG'), right, IERR )
        if (ierr == 0) then
            read (right, *) kerk_amix_mag
        else
            kerk_amix_mag=1.0d0
        ENDIF
        if(inode_tot.eq.1) then
            write(22,*) "KERK_AMIX_MAG = ", kerk_amix_mag
        endif

        CALL read_key_words ( 9, 'KERK_BMIX', LEN('KERK_BMIX'), right, IERR )
        if (ierr == 0) then
            read (right, *) kerk_bmix
        else
            kerk_bmix=0.5d0
        ENDIF
        if(inode_tot.eq.1) then
            write(22,*) "KERK_BMIX = ", kerk_bmix
        endif

        CALL read_key_words ( 9, 'LDAU_MIX', LEN('LDAU_MIX'), right, IERR )
        if (ierr == 0) then
            read (right, *) ldau_mix
        else
            ldau_mix=0.7d0
        ENDIF
        if(inode_tot.eq.1) then
            write(22,*) "LDAU_MIX = ", ldau_mix
        endif

        CALL read_key_words ( 9, 'PULAY_WEIGHT_SPIN', LEN('PULAY_WEIGHT_SPIN'), right, IERR )
        if (ierr == 0) then
            read (right, *) pulay_weight_spin
        else
            pulay_weight_spin=1.d0
            !if(is_LDAU) then
            !    pulay_weight_spin=1.d0
            !endif
        ENDIF
        if(inode_tot.eq.1) then
            write(22,*) "PULAY_WEIGHT_SPIN = ", pulay_weight_spin
        endif

        CALL read_key_words ( 9, 'PULAY_WEIGHT_NS', LEN('PULAY_WEIGHT_NS'), right, IERR )
        if (ierr == 0) then
            read (right, *) pulay_weight_ns
        else
            pulay_weight_ns=1.d0
        ENDIF
        if(inode_tot.eq.1) then
            write(22,*) "PULAY_WEIGHT_NS = ", pulay_weight_ns
        endif 
        !
        CALL read_key_words ( 9, 'OUT.MLMD', LEN('OUT.MLMD'), right, IERR )
        if(ierr.eq.0) then 
            read ( right, * ) MCTRL_OUT_MLMD
        else
            MCTRL_OUT_MLMD=.false.
        endif
        if(inode_tot.eq.1) then
            write(22,*) "OUT.MLMD    = ", MCTRL_OUT_MLMD
        endif
        !============================================================
        if(inode_tot.eq.1) write(22,1455) numMPIPerGPU 
        1455   format(" NUM_MPI_PER_GPU = ",i5)

        if(inode_tot .eq. 1) then
          write(*,*) ' ===================================== '
          write(*,*) numMPIPerGPU, 'ranks per GPU'
          write(*,*) ' ===================================== '
        endif
        !============================================================
        
        CALL read_key_words(9,'FLAG_CYLINDER',LEN('FLAG_CYLINDER'),right,IERR )
        if(ierr.eq.0) then
            READ ( right, * ) iflag_cylinder
        else
            iflag_cylinder=0
        endif

        if(inode_tot.eq.1) then
            write(22,*) "FLAG_CYLINDER = ", iflag_cylinder
        endif

        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc 
        CALL read_key_words( 9, 'SMOOTH_VLOC', LEN('SMOOTH_VLOC'), right, IERR)
        if ( ierr .eq. 0) then
            READ( right, * ) smooth_vloc
        else
            smooth_vloc = .true.
        endif
        ! do not output the smooth_vloc, used for test.
        !if(inode_tot .eq. 1) write(22,*) "SMOOTH_VLOC=",smooth_vloc
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc 
        CALL read_key_words( 9, 'USE_PWSCF_INTE_METHOD', LEN('USE_PWSCF_INTE_METHOD'), right, IERR)
        if ( ierr .eq. 0) then
            READ( right, * ) pwscf_integration_method
        else
            pwscf_integration_method=.false.
        endif
        if(PWSCF_OUTPUT) then
            pwscf_integration_method=.true.
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc 
        ! infermi=.false.
        ioc_rot=.false.
        !if(iproj_occ.eq.2.or.iproj_occ.eq.3) inFermi=.true.

        do i=1,niter0
        !    if(itypeFermi0(i).eq.0.or.itypeFermi0(i).eq.-1) then
        !        inFermi=.true.
        !    endif
            if(itypeFermi0(i).eq.-1) then
                icc_rot=.true.
            endif
        enddo
        do i=1,niter1
        !    if(itypeFermi1(i).eq.0.or.itypeFermi1(i).eq.-1) then
        !        inFermi=.true.
        !    endif
            if(itypeFermi1(i).eq.-1) then
                icc_rot=.true.
            endif
        enddo

        !if(iproj_occ.eq.3) infermi=.true.

        !if ( inFermi ) then



        if (iproj_occ.ne.-1.or.itddft_nob.eq.1) then      ! the case for IN.OCC=T
              if(itddft_nob.eq.1) then
                iproj_occ_old=iproj_occ
                iproj_occ=3
               endif
                    if(iproj_occ.eq.3) then
                    m_proj_sp(1)=mx
                    m_proj_sp(2)=mx
                    endif
            call init_smearing(mx,nkpt)
            call init_mod_map_proj_occ(mx,nkpt,islda)
              if(itddft_nob.eq.1) then
                iproj_occ=iproj_occ_old
              endif
        endif
        


        if (iproj_occ.ne.-1) then      ! the case for IN.OCC=T

            do ispin=1,islda 
                if(ispin .eq. 1) then
                    open(10,file="IN.OCC",status='old',action='read',iostat=ierr)
                    if(ierr.ne.0) then
                        write(message,*) "OCCUP FILE IN.OCC does not exist, stop"
                        call error_stop(message,__FILE__,__LINE__)
                    endif
                    rewind(10)
                    do ikpt=1,nkpt
                        if(iproj_occ.ne.2) then
                          read(10,*) (ferup(iband,ikpt),iband=1,mx)
                          do iband=1,mx

                          iferup_iproj3(iband,ikpt)=0
                          if(ferup(iband,ikpt).lt.0.d0) then
                            ferup(iband,ikpt)=-ferup(iband,ikpt)
                            iferup_iproj3(iband,ikpt)=1
                          endif

                          if(ferup(iband,ikpt).lt.0.d0.or.ferup(iband,ikpt).gt.1.d0) then
                            write(6,*) "in IN.OCC, the occup must be between [0,1],stop"
                            stop
                          endif
                          enddo

                        else
                            read(10,*)
                        endif
                        !read(10,*)
                    enddo ! ikpt

                    if(iproj_occ.eq.3) then
                      add_charge_occ2=0.d0
                      do ikpt=1,nkpt
                      do m=1,mx
                      imap_proj_sp(m,ikpt,1)=m
                      imap_proj(m,ikpt,1)=m
                      occ_proj_sp(m,ikpt,1)=abs(ferup(m,ikpt))*weighkpt_2(ikpt)*2.d0/islda
                      add_charge_occ2=add_charge_occ2+occ_proj_sp(m,ikpt,1)
                      enddo
                      enddo
                    endif



                    if(iproj_occ.eq.2) then
                        read(10,*) 
                        read(10,*) m_proj_sp(1)
                     if(m_proj_sp(1).gt.10) then
                     write(6,*) "m_proj_sp cannot be larger than10,IN.OCC,iproj=2"
                     stop
                     endif
                        add_charge_occ2=0.d0
                        do ikpt=1,nkpt
                            do m1=1,m_proj_sp(1)
                                read(10,*) imap_proj_sp(m1,ikpt,1),occ_proj_sp(m1,ikpt,1)
                                occ_proj_sp(m1,ikpt,1)=occ_proj_sp(m1,ikpt,1)*weighkpt_2(ikpt)*2.d0/islda
                                add_charge_occ2=add_charge_occ2+occ_proj_sp(m1,ikpt,1)
                            enddo
                        enddo
                    endif
                    close(10)
                endif

                if(ispin .eq. 2) then
                    open(10,file="IN.OCC_2",status='old',action='read',iostat=ierr)
                    if(ierr.ne.0) then
                        write(message,*) "OCCUP FILE IN.OCC_2 does not exist, stop"
                        call error_stop(message,__FILE__,__LINE__)
                    endif
                    rewind(10)
                    do ikpt=1,nkpt
                        if(iproj_occ.ne.2) then
                            read(10,*) (ferdw(iband,ikpt),iband=1,mx)
                            do iband=1,mx

                          iferdw_iproj3(iband,ikpt)=0
                          if(ferdw(iband,ikpt).lt.0.d0) then
                            ferdw(iband,ikpt)=-ferdw(iband,ikpt)
                            iferdw_iproj3(iband,ikpt)=1
                          endif

                            if(ferdw(iband,ikpt).lt.0.d0.or.ferup(iband,ikpt).gt.1.d0) then
                              write(6,*) "in IN.OCC_2, the occup must be between [0,1],stop"
                              stop
                            endif
                            enddo

                          else
                            read(10,*) 
                        endif
                        !read(10,*)
                    enddo

                    if(iproj_occ.eq.3) then

                      add_charge_occ2=0.d0
                      do ikpt=1,nkpt
                      do m=1,mx
                      imap_proj_sp(m,ikpt,2)=m
                      imap_proj(m,ikpt,2)=m
                      occ_proj_sp(m,ikpt,2)=abs(ferdw(m,ikpt))*weighkpt_2(ikpt)*2.d0/islda
                      add_charge_occ2=add_charge_occ2+occ_proj_sp(m,ikpt,2)
                      enddo
                      enddo
                    endif


                    if(iproj_occ.eq.2) then
                        read(10,*) 
                        read(10,*) m_proj_sp(2)
                     if(m_proj_sp(2).gt.10) then
                     write(6,*) "m_proj_sp cannot be larger than10,IN.OCC,iproj=2"
                     stop
                     endif
                        do ikpt=1,nkpt
                            do m1=1,m_proj_sp(2)
                                read(10,*) imap_proj_sp(m1,ikpt,2),occ_proj_sp(m1,ikpt,2)
                                occ_proj_sp(m1,ikpt,2)=occ_proj_sp(m1,ikpt,2)*weighkpt_2(ikpt)*2.d0/islda
                                add_charge_occ2=add_charge_occ2+occ_proj_sp(m1,ikpt,2)
                            enddo
                        enddo
                    endif
                    close(10)
                endif
            enddo
            !read

        endif

        if ( icc_rot ) then
            allocate(cc_rot(mx,mx,nkpt,islda))
            allocate(ind_tmp(mx))
            allocate(weigh_tmp(mx))
            cc_rot=cmplx(0.d0,0.d0)
            do ispin=1,islda 
                if(ispin .eq. 1) then
                    open(10,file="IN.CC",status='old',action='read',iostat=ierr)
                    if(ierr.ne.0) then
                        write(message,*) "OCCUP FILE IN.CC does not exist, stop"
                        call error_stop(message,__FILE__,__LINE__)
                    endif
                    rewind(10)
                    do ikpt=1,nkpt
                        do im1=1,mx
                            read(10,*) ntmp,((ind_tmp(im2),weigh_tmp(im2)),im2=1,ntmp)
                            do im2=1,ntmp
                                cc_rot(ind_tmp(im2),im1,ikpt,ispin)=dsqrt(abs(weigh_tmp(im2)))
                            enddo
                        enddo
                    enddo
                    close(10)
                endif
                if(ispin .eq. 2) then
                    open(10,file="IN.CC_2",status='old',action='read',iostat=ierr)
                    if(ierr.ne.0) then
                        write(message,*) "OCCUP FILE IN.CC_2 does not exist, stop"
                        call error_stop(message,__FILE__,__LINE__)
                    endif
                    rewind(10)
                    do ikpt=1,nkpt
                        do im1=1,mx
                            read(10,*) ntmp,((ind_tmp(im2),weigh_tmp(im2)),im2=1,ntmp)
                            do im2=1,ntmp
                                cc_rot(ind_tmp(im2),im1,ikpt,ispin)=dsqrt(abs(weigh_tmp(im2)))
                            enddo
                        enddo
                    enddo
                    close(10)
                endif
            enddo
            deallocate(ind_tmp)
            deallocate(weigh_tmp)
            do ispin=1,islda
                do ikpt=1,nkpt
                    do im1=1,mx
                        do im2=1,im1-1
                            cc=cmplx(0.d0,0.d0)
                            do im3=1,mx
                                cc=cc+cc_rot(im3,im1,ikpt,ispin)*conjg(cc_rot(im3,im2,ikpt,ispin))
                            enddo
                            do im3=1,mx
                                cc_rot(im3,im1,ikpt,ispin)=cc_rot(im3,im1,ikpt,ispin)-cc*cc_rot(im3,im2,ikpt,ispin)
                            enddo
                        enddo ! im2
                        sum=0.d0
                        do im3=1,mx
                            sum=sum+abs(cc_rot(im3,im1,ikpt,ispin))**2
                        enddo
                        sum=1.d0/dsqrt(sum)
                        do im3=1,mx
                            cc_rot(im3,im1,ikpt,ispin)=sum*cc_rot(im3,im1,ikpt,ispin)
                        enddo
                    enddo
                enddo
            enddo
        endif

        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc 
        allocate(occ_smearing(mx,nkpt,islda))
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc 
        if(inode_tot.eq.1) then
            write(22,*)
            write(22,*) "------------------------------------------------"
            write(22,*) "total number of K-point:", nkpt
            do i = 1, nkpt
                write(22,2016) akx_2(i), aky_2(i), akz_2(i), weighkpt_2(i)
            enddo
            2016   format(f10.5,2x,f10.5,1x,f10.5,2x,f10.5)
            write(22,*) "------------------------------------------------"
            write(22,*) 
            write(22,*) 
            write(22,*) "*********************************************"
            write(22,*) "*********** end of etot.input report ********"
            fackpt=dsqrt(2.d0*Ecut2)/(4*datan(1.d0))
            dd11=fackpt*dsqrt(AL(1,1)**2+AL(2,1)**2+AL(3,1)**2)
            dd22=fackpt*dsqrt(AL(1,2)**2+AL(2,2)**2+AL(3,2)**2)
            dd33=fackpt*dsqrt(AL(1,3)**2+AL(2,3)**2+AL(3,3)**2)
            write(22,*) "minimum n1,n2,n3 from Ecut2"
            write(22,1122) dd11,dd22,dd33
            fackpt=dsqrt(2.d0*Ecut2L)/(4*datan(1.d0))
            dd11=fackpt*dsqrt(AL(1,1)**2+AL(2,1)**2+AL(3,1)**2)
            dd22=fackpt*dsqrt(AL(1,2)**2+AL(2,2)**2+AL(3,2)**2)
            dd33=fackpt*dsqrt(AL(1,3)**2+AL(2,3)**2+AL(3,3)**2)
            write(22,*) "minimum n1L,n2L,n3L from Ecut2L"
            write(22,1122) dd11,dd22,dd33
            1122      format(3(f12.3,1x))
            write(22,*) "*********************************************"
            write(22,*) 
            write(22,*) 
        endif


        close(9)
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !       write(6,*) niter0,nline0
        !       write(6,*) TOTNEL,mix
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        iflag=0
        if(irho_out.eq.2) iflag=iflag+1
        if(ivr_out.eq.2) iflag=iflag+1
        if(idens_out.eq.2.or.idens_out.eq.21.or.idens_out.eq.22) iflag=iflag+1
        if(iflag.gt.1) then
            write(message,*)"irho_out,ivr_out,idens_out cannot have more than one 2",irho_out,ivr_out,idens_out
            call error_stop(message,__FILE__,__LINE__)
        endif

        if(n1.eq.n1L.and.n2.eq.n2L.and.n3.eq.n3L.and.dabs(Ecut2-Ecut2L).lt.0.01) then
            iflag_fft2L=0        ! fft2 and fft2L are the same
        else
            iflag_fft2L=1
        endif

        !*************************************************

        if(isym.eq.0) then
            nrot=1
            frac_sl=0.d0
            smatr=0.d0
            do i=1,48
                smatr(1,1,i)=1.d0
                smatr(2,2,i)=1.d0
                smatr(3,3,i)=1.d0    ! LWW, July 2017
            enddo
        else
            open(10,file=sym_file,status='old',action='read',iostat=ierr)
            if(ierr.ne.0) then
                if(inode.eq.1)write(message,*) "sym_file ***", sym_file, "*** does not exist, stop"
                call error_stop(message,__FILE__,__LINE__)
            endif
            rewind(10)
            frac_sl = 0.0d0
            read(10,*) nrot
            do irot=1,nrot
                read(10,*)
                do j=1,3
                    read(10,*) smatr(1,j,irot),smatr(2,j,irot),smatr(3,j,irot)
                enddo
                read(10,*) frac_sl(1:3,irot)  !new
            enddo
            close(10)
        endif

        nr=n1*n2*n3

        if(inode_tot.eq.1) then
            write(6,*) "number of nonlocal atom=", natom
            write(6,*) "n1,n2,n3=", n1,n2,n3
            write(6,*) "n1L,n2L,n3L=", n1L,n2L,n3L
            !write(6,*) "SPIN, XcFunctional=", islda, igga
            write(6,*) "SPIN=",islda
            write(6,*) "AL1,AL2,AL3 in (x,y,z) components and A"
            write(6,*) "each line is one supercell edge vector"
            write(6,3) A_AU_1*AL(1,1),A_AU_1*AL(2,1),A_AU_1*AL(3,1)
            write(6,3) A_AU_1*AL(1,2),A_AU_1*AL(2,2),A_AU_1*AL(3,2)
            write(6,3) A_AU_1*AL(1,3),A_AU_1*AL(2,3),A_AU_1*AL(3,3)
        endif

        3      format(3(f13.7,1x))
        !*************************************************
        nh1=n1/2+1

        ! hse hse.
        if(use_hse .eq. 1 .or. interp_DOS_999.eq.1 ) then
          call read_sx(AL,nkpt,nrot,smatr)
         endif
        !*************************************************
        
        return
    contains
        subroutine old_fashion_xatom_assign(ac)
                implicit none
                type(atom_config) :: ac
                !
                natom=ac.natom
                !
                AL=ac.AL_bohr
                !
                stress_mask=ac.stress_mask
                !
                stress_ext=ac.stress_ext
                !
                iatom(1:natom)=ac.tatom(1:natom)
                !
                xatom(1:3,1:natom)=ac.xatom(1:3,1:natom)
                !
                imov_at(1:3,1:natom)=ac.imov(1:3,1:natom)
                !
                if (.not. allocated(is_gupta)) allocate(is_gupta(natom))
                is_gupta(1:natom)=ac.flag1(1:natom) 
                !
                VX_1(1:natom)=ac.velocity(1,1:natom)
                VY_1(1:natom)=ac.velocity(2,1:natom)
                VZ_1(1:natom)=ac.velocity(3,1:natom)
                !
                langevin_factT(1:natom)=ac.langevin_factT(1:natom)
                langevin_factG(1:natom)=ac.langevin_factG(1:natom)
                !
                constraint_mag_atom(1:natom)=ac.constraint_mag_atom(1:natom)
                constraint_mag_alpha(1:natom)=ac.constraint_mag_alpha(1:natom)
                !
                weight_mag(1:natom)=ac.weight_mag_frac(1:natom)
                !
                weight_magxyz(1:natom,1)=ac.weight_mag_xyz_frac(1,1:natom)
                weight_magxyz(1:natom,2)=ac.weight_mag_xyz_frac(2,1:natom)
                weight_magxyz(1:natom,3)=ac.weight_mag_xyz_frac(3,1:natom)
        end subroutine old_fashion_xatom_assign
        !*******************************************
        subroutine readvwr_head()

                implicit double precision (a-h,o-z)

                open(10,file=vwr_atom(ia),status='old',action='read',iostat=ierr)
                if(ierr.ne.0) then
                    if(inode.eq.1)write(message,*) "vwr_file ***",filename,"*** does not exist, stop"
                    call error_stop(message,__FILE__,__LINE__)
                endif
                read(10,*) nrr_t,ic_t,iiatom(ia),zatom_t,iloc_t,occ_s_t,occ_p_t,occ_d_t,iso,Ecut_t,amass_t
                read(10,*) is_ref_t,ip_ref_t,id_ref_t,is_TB_t,ip_TB_t,id_TB_t
                close(10)

                if(iloc_t.eq.1) is_ref_t=0
                if(iloc_t.eq.2) ip_ref_t=0
                if(iloc_t.eq.3) id_ref_t=0

                !ccccccccccccccccccccccccccccccccccccccccccccccccccccc
                if(ido_DOS_999.eq.1) then
                    is_ref_t=1
                    ip_ref_t=1
                    id_ref_t=1
                endif

                !cccccccccccccccccccccccccccccccccccccccccccccccccccccccc

                nref_type(ia)=is_ref_t+ip_ref_t*3+id_ref_t*5
                z_type(ia) = zatom_t
                Ecut_type(ia)= Ecut_t
                amass_type(ia) = amass_t

                return
        end subroutine readvwr_head



        subroutine readf_xatom()
                implicit double precision (a-h,o-z)
                real*8, allocatable, dimension(:,:) :: xatom_tmp
                integer, allocatable, dimension(:,:) :: imov_latt_vectmp
                integer, allocatable, dimension(:)   ::  iatom_tmp

                open(10,file=f_xatom,status='old',action='read',iostat=ierr)

                !       write(*,*)"f_xatom",f_xatom
                !       stop

                if(ierr.ne.0) then
                    if(inode.eq.1) write(message,*) "IN.ATOM ***",f_xatom, " ***does not exist, stop"
                    call error_stop(message,__FILE__,__LINE__)
                endif

                rewind(10)
                read(10,*) natom

                if(natom.gt.matom_1) then
                    if(inode.eq.1) then
                        write(message,*) "natom.gt.matom_1, increase matom_1 in data.f, stop",f_xatom,natom,matom_1
                        call error_stop(message,__FILE__,__LINE__)
                    endif
                endif

                allocate(xatom_tmp(3,natom))
                allocate(imov_latt_vectmp(3,natom))
                allocate(iatom_tmp(natom))

                read(10,*) (AL(i,1),i=1,3)
                read(10,*) (AL(i,2),i=1,3)
                read(10,*) (AL(i,3),i=1,3)

                AL = AL/A_AU_1 !zhilin

                do i=1,natom
                    read(10,*) iatom_tmp(i),xatom_tmp(1,i),xatom_tmp(2,i),xatom_tmp(3,i),imov_latt_vectmp(1,i),imov_latt_vectmp(2,i),imov_latt_vectmp(3,i)
                enddo
                close(10)

                !ccccccccccccccccccccccccccccccccccccccccccccccccccc
                !ccc Now, re-arrange xatom, so the same atoms are consequentive together. 
                !ccc This is useful to speed up the getwmask.f
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

                ii=0
                100    continue
                ncount=0
                itype_tmp=-2
                do i=1,natom
                    if(itype_tmp.eq.-2.and.iatom_tmp(i).ne.-1)itype_tmp=iatom_tmp(i)

                    if(iatom_tmp(i).eq.itype_tmp) then
                        ii=ii+1
                        ncount=ncount+1
                        iatom(ii)=iatom_tmp(i)
                        iatom_tmp(i)=-1
                        xatom(:,ii)=xatom_tmp(:,i)
                        imov_at(:,ii)=imov_latt_vectmp(:,i)
                    endif
                enddo
                if(ncount.gt.0) goto 100
                if(ii.ne.natom) then
                    write(message,*) "something wrong to rearrange xatom, stop"
                    call error_stop(message,__FILE__,__LINE__)
                endif

                deallocate(xatom_tmp)
                deallocate(imov_latt_vectmp)
                deallocate(iatom_tmp)

                return
        end subroutine readf_xatom
        !**************************************************


        subroutine readkpt()

                implicit double precision (a-h,o-z)
                real*8 AL_t(3,3),tmp(3)

                if(ikpt_yno_999.eq.0)  then 
                    nkpt=1
                else
                    open(12,file=kpt_file,status='old',action='read',iostat=ierr)
                    if(ierr.ne.0) then
                        if(inode.eq.1) write(message,*) "Kpoint file***",kpt_file,"***does not exist, stop"
                        call error_stop(message,__FILE__,__LINE__)
                    endif
                    rewind(12)
                    read(12,*) nkpt
                endif

                if(nkpt .gt. 8000) then
                    if(inode_tot .eq. 1) then 
                        write(message,*) "Too many K-points for PWmat! upperlimit is 8000 K-points!"
                        call error_stop(message,__FILE__,__LINE__)
                    endif
                endif
                call data_allocate_akx(nkpt)
                call data_allocate_akx0(nkpt)
                call ngtot_allocate_9(nnodes,nkpt)
                call ngtot_allocate_10(nnodes,nkpt)
                call ngtot_allocate_11(nnodes,nkpt)

                if(ikpt_yno_999.eq.0) then
                    akx_2(1)=0.d0
                    aky_2(1)=0.d0
                    akz_2(1)=0.d0
                    akx0_2(1)=0.d0
                    aky0_2(1)=0.d0
                    akz0_2(1)=0.d0
                    weighkpt_2(1)=1.d0
                    return
                else

                    pi=4*datan(1.d0)

                    read(12,*) iflag, ALxyz

                    if(iflag.eq.1)  then
                        if(inode_tot.eq.1) write(6,*) "input kpts in Cartesian Coord"
                        sumw=0.d0
                        do kpt=1,nkpt
                            read(12,*) akx_2(kpt),aky_2(kpt),akz_2(kpt),weighkpt_2(kpt)
                            sumw=sumw+weighkpt_2(kpt)
                            akx_2(kpt)=akx_2(kpt)*2*pi/ALxyz
                            aky_2(kpt)=aky_2(kpt)*2*pi/ALxyz
                            akz_2(kpt)=akz_2(kpt)*2*pi/ALxyz
                            akx0_2(kpt)=akx_2(kpt)
                            aky0_2(kpt)=aky_2(kpt)
                            akz0_2(kpt)=akz_2(kpt)
                        enddo
                    endif

                    if(iflag.eq.2) then
                        if(inode_tot.eq.1) write(6,*) "input kpts in primary cell unit"

                        do i=1,3
                            do j=1,3
                                AL_t(j,i)=AL(i,j)
                            enddo
                            tmp(i)=1
                        enddo
                        call gaussj(AL_t,3,3,tmp,1,1)

                        sumw=0.d0
                        do kpt=1,nkpt
                            read(12,*) ak1_t,ak2_t,ak3_t,weighkpt_2(kpt)
                            sumw=sumw+weighkpt_2(kpt)
                            akx_2(kpt)=2*pi*(AL_t(1,1)*ak1_t+AL_t(1,2)*ak2_t+AL_t(1,3)*ak3_t)
                            aky_2(kpt)=2*pi*(AL_t(2,1)*ak1_t+AL_t(2,2)*ak2_t+AL_t(2,3)*ak3_t)
                            akz_2(kpt)=2*pi*(AL_t(3,1)*ak1_t+AL_t(3,2)*ak2_t+AL_t(3,3)*ak3_t)
                            akx0_2(kpt)=akx_2(kpt)
                            aky0_2(kpt)=aky_2(kpt)
                            akz0_2(kpt)=akz_2(kpt)
                        enddo
                    endif

                    close(12)

                endif


                if(dabs(sumw-1.d0).gt.0.0000001d0) then
                    if(inode.eq.1) then
                        write(6,*) "**** Warning, sum of kpt weights not eq. 1", sumw
                        write(6,*) "**** renormalize the kpt weight to 1"
                    endif
                    do kpt=1,nkpt
                        weighkpt_2(kpt)=weighkpt_2(kpt)/sumw
                    enddo
                endif

                return
        end subroutine readkpt

        subroutine get_n123(n_t,dd,start)
                implicit double precision(a-h,o-z)
                real*8 dd
                integer n_t
                integer :: start
                external :: nnn
                !          integer nnn(2000)   ! FFT dimension table
                ntot_tmp=614       !  to be completed later
                do i=start,ntot_tmp
                    if(nnn(i-1).le.dd.and.dd.lt.nnn(i)) then
                        n_t=nnn(i)
                        start = i
                        exit
                    endif
                enddo
                !      if(iflag .eq. 1) then
                !      write(*,*) 'iflag-----------------------', iflag
                !      do i=i, ntot_tmp
                !       n_t = nnn(i)
                !       if(mod(n_t, nnodes_tot) .eq. 0) goto 133
                !      enddo
                !      write(*,*) "N123 value is wrong!", n_t
                !      call mpi_abort(mpi_comm_world, ierr)
                !      endif
                !133   continue
                return
        end subroutine get_n123


        subroutine get_n456(n_t)
                implicit double precision(a-h,o-z)
                integer n_t
                external :: nnn
                !          integer nnn(2000)   ! FFT dimension table
                ntot_tmp=46 !  to be completed later
                do i=2,ntot_tmp
                    if(nnn(i-1).le.n_t.and.n_t.lt.nnn(i)) then
                        n_t=nnn(i)
                        goto 123
                    endif
                enddo
                123   continue
                return

        end subroutine get_n456

        !*******************************************************
        subroutine readf_xatom_new()
                implicit double precision (a-h,o-z)
                real*8, allocatable, dimension(:,:) :: xatom_tmp
                integer, allocatable, dimension(:,:) :: imov_latt_vectmp
                integer, allocatable, dimension(:)   ::  iatom_tmp,iatom_tmp2
                real*8, allocatable, dimension(:) :: VX_1_tmp,VY_1_tmp,VZ_1_tmp
                real*8, allocatable, dimension(:) :: weight_mag_tmp
                real*8, allocatable, dimension(:,:) :: weight_mag_tmpxyz
                real*8, allocatable, dimension(:) ::langevin_factT_tmp,langevin_factG_tmp
                real*8, allocatable, dimension(:) :: constraint_mag_atom_tmp
                real*8, allocatable, dimension(:) :: constraint_mag_alpha_tmp
                real*8, allocatable, dimension(:,:) :: LDAU_lambda_tmp
                integer, allocatable, dimension(:)   :: ind_order_old2new_tmp,ind_order_new2old_tmp
                real*8, allocatable, dimension(:) :: weight_atom_tmp

                logical :: scanit
                intrinsic :: sum

                open(10,file=f_xatom,status='old',action='read',iostat=ierr)

                if(ierr.ne.0) then
                    if(inode.eq.1) write(message,*) "IN.ATOM",f_xatom,"not exist, stop"
                    call error_stop(message,__FILE__,__LINE__)
                endif

                rewind(10)
                read(10,*) natom
                if(natom.gt.matom_1) then
                    if(inode.eq.1) then
                        write(message,*) "natom.gt.matom_1, increase matom_1 in data.f,stop ",f_xatom,natom,matom_1
                        call error_stop(message,__FILE__,__LINE__) 
                    endif
                endif

                allocate(xatom_tmp(3,natom))
                allocate(imov_latt_vectmp(3,natom))
                allocate(iatom_tmp(natom))
                allocate(iatom_tmp2(natom))
                allocate(VX_1_tmp(natom))
                allocate(VY_1_tmp(natom))
                allocate(VZ_1_tmp(natom))
                allocate(weight_mag_tmp(natom))
                allocate(weight_mag_tmpxyz(natom,3))
                allocate(langevin_factT_tmp(natom))
                allocate(langevin_factG_tmp(natom))
                allocate(constraint_mag_atom_tmp(natom))
                allocate(constraint_mag_alpha_tmp(natom))
                allocate(ind_order_old2new_tmp(natom))
                allocate(ind_order_new2old_tmp(natom))
                allocate(LDAU_lambda_tmp(natom,2))
                allocate(weight_atom_tmp(natom))
                allocate(weight_atom_1(natom))


                if (.not. allocated(is_gupta)) allocate(is_gupta(natom))
                is_gupta = 0

                call scan_key_words (10, "LATTICE", len("LATTICE"), scanit)
                if(scanit) then
                    read(10,*) (AL(i,1),i=1,3)
                    read(10,*) (AL(i,2),i=1,3)
                    read(10,*) (AL(i,3),i=1,3)
                else
                    write(message,*)"Must provide LATTICE in IN.ATOM file ",ADJUSTL(trim(f_xatom)) 
                    call error_stop(message,__FILE__,__LINE__)
                endif

                AL = AL/A_AU_1 !zhilin

                call scan_key_words (10, "STRESS_MASK", len("STRESS_MASK"), scanit)
                if(.not. scanit) then
                    stress_mask=1
                else
                    read(10,*) (stress_mask(i,1),i=1,3)
                    read(10,*) (stress_mask(i,2),i=1,3)
                    read(10,*) (stress_mask(i,3),i=1,3)
                endif
                call scan_key_words (10, "STRESS_EXTERNAL", len("STRESS_EXTERNAL"), scanit)
                if(.not. scanit) then
                    exist_stress_ext=.false.
                else
                    exist_stress_ext=.true.
                    read(10,*) (stress_ext(i,1),i=1,3)
                    read(10,*) (stress_ext(i,2),i=1,3)
                    read(10,*) (stress_ext(i,3),i=1,3)
                    stress_ext=stress_ext/Hartree_ev*natom
                endif

                call scan_key_words (10, "POSITION", len("POSITION"), scanit)
                if(.not. scanit) then
                    write(*,*) "keyword 'position' is needed at",f_xatom
                    if(inode_tot .eq. 1) then
                        call mpi_abort(mpi_comm_world,ierr)
                    endif
                else
                    if (irmethod_1 == 4) then
                        do i=1,natom
                            read(10,*) iatom_tmp(i),xatom_tmp(1,i),xatom_tmp(2,i),xatom_tmp(3,i),imov_latt_vectmp(1,i),imov_latt_vectmp(2,i),imov_latt_vectmp(3,i), is_gupta(i), is_gupta(i)
                        end do
                    else
                        do i=1,natom
                            read(10,*) iatom_tmp(i),xatom_tmp(1,i),xatom_tmp(2,i),xatom_tmp(3,i),imov_latt_vectmp(1,i),imov_latt_vectmp(2,i),imov_latt_vectmp(3,i)
                        end do
                    end if
                    !!!!!  check imov avoiding imov==0 in POSITION-RELAX MD NEB TDDFT NAMD
                    imov_sum = sum(imov_latt_vectmp)
                    select case (trim(adjustl(JOB_1)))
                    case ("MD","NAMD","NEB")
                        if (imov_sum == 0) then
                            write (6, *) "atoms not setting move, please check atom.config for JOB = "//trim(JOB_1)
                            call mpi_abort(mpi_comm_world,ierr)
                        end if
                        !case ("RELAX")
                        !    if (imov_sum == 0 .and. istress_cal == 0) then
                        !        write (6, *) "atoms not setting move, please check atom.config for JOB = "//trim(JOB_1)
                        !        call mpi_abort(mpi_comm_world,ierr)
                        !    end if
                    case default
                    end select
                    !!!!!
                endif


                call scan_key_words (10, "VELOCITY", len("VELOCITY"), scanit)
                if(.not. scanit) then
                    !write(*,*) "keyword 'velocity' is needed at",f_xatom
                    !stop
                    if ((iMD .eq. 11) .or.(iMD .eq. 22) .or.(iMD .eq. 33)) then
                        write(*,*) "keyword 'velocity' is needed at",f_xatom
                        if(inode_tot .eq. 1) call mpi_abort(mpi_comm_world,ierr)
                    endif
                    exist_velocity=.false.
                endif
                if(scanit) then
                    if(inode_tot .eq. 1) then
                        write(*,*) "velocity read in from ", f_xatom 
                    endif
                    do i=1,natom
                        read(10, *) iatom_tmp2(i),VX_1_tmp(i),VY_1_tmp(i),VZ_1_tmp(i)
                        if(iatom_tmp2(i).ne.iatom_tmp(i)) then
                            write(6,*) "order of iatom in position/velocity not the same",i
                            if(inode_tot .eq. 1) then
                                call mpi_abort(mpi_comm_world,ierr)
                            endif
                        endif
                    enddo
                    exist_velocity=.true.
                endif

                call scan_key_words (10, "LANGEVIN_ATOMFACT_TG", len("LANGEVIN_ATOMFACT_TG"), scanit)
                if(.not. scanit) then

                    do i=1,natom
                        langevin_factT_tmp(i)=1.d0
                        langevin_factG_tmp(i)=1.d0
                    enddo
                else
                    do i=1,natom
                        read(10, *) iatom_tmp2(i),langevin_factT_tmp(i),langevin_factG_tmp(i)
                    enddo
                endif

                call scan_key_words (10, "CONSTRAINT_MAG", len("CONSTRAINT_MAG"), scanit)
                if(.not. scanit) then
                    do i=1,natom
                        constraint_mag_atom_tmp(i)=0.d0
                        constraint_mag_alpha_tmp(i)=0.d0
                    enddo
                else
                    do i=1,natom
                        read(10, *) iatom_tmp2(i),constraint_mag_atom_tmp(i),constraint_mag_alpha_tmp(i)
                    enddo
                endif


                call scan_key_words (10, "MAGNETIC", len("MAGNETIC"), scanit)
                if(.not. scanit) then
                    do i=1,natom
                        !weight_mag_tmp(i)=0.2     ! Note, the meanging has been changed, to be align with Magentic_XYZ 
                        weight_mag_tmp(i)=1.0d0     ! Note, the meanging has been changed, to be align with Magentic_XYZ 
                        do j = 1, ntype
                            if (iatom_tmp(i) == upfpsp(j)%num) weight_mag_tmp(i) = weight_mag_tmp(i) / upfpsp(j)%zp
                        end do 
                    enddo
                else
                    if(inode_tot .eq. 1) then
                        write(*,*) "initial magnetic percentage read in from ", f_xatom 
                        WRITE(6, *) "***WARNING***: Atomic Magnet read from ",f_xatom
                        WRITE(6, *) "IN.SYMM does not take into account Magnet"
                        WRITE(6, *) "Need to use proper IN.SYMM to ensure AFM etc"
                    endif
                    do i=1,natom
                        read(10, *) iatom_tmp2(i), weight_mag_tmp(i)
                        do j = 1, ntype
                            if (iatom_tmp2(i) == upfpsp(j)%num) weight_mag_tmp(i) = weight_mag_tmp(i) / upfpsp(j)%zp
                        end do 
                        if(iatom_tmp2(i).ne.iatom_tmp(i)) then
                            write(6,*) "order of iatom in position/magnetic not the same", i
                            if(inode_tot .eq. 1) then
                                call mpi_abort(mpi_comm_world,ierr)
                            endif
                        endif
                    enddo
                endif

                call scan_key_words (10, "MAGNETIC_XYZ", len("MAGNETIC_XYZ"), scanit)
                if(.not. scanit) then
                    do i=1,natom
                        !weight_mag_tmpxyz(i,1)=0.0d0
                        !weight_mag_tmpxyz(i,2)=0.0d0
                        !weight_mag_tmpxyz(i,3)=0.2d0
                        weight_mag_tmpxyz(i,1)=1.0d0
                        weight_mag_tmpxyz(i,2)=1.0d0
                        weight_mag_tmpxyz(i,3)=1.0d0
                        do j = 1, ntype
                            if (iatom_tmp(i) == upfpsp(j)%num) weight_mag_tmpxyz(i,1:3) = weight_mag_tmpxyz(i,1:3) / upfpsp(j)%zp
                        end do
                    enddo
                else
                    if(inode_tot .eq. 1) then
                        write(*,*)  "initial spin x,y,z percentage read in from ", f_xatom 
                        WRITE(6, *) "***WARNING***: Atomic Magnet read from ",f_xatom
                        WRITE(6, *) "IN.SYMM does not take into account Magnet"
                        WRITE(6, *) " IN.SYMM does not work for noncollinear magnetic"
                    endif

                    do i=1,natom
                        read(10, *) iatom_tmp2(i), weight_mag_tmpxyz(i,1), weight_mag_tmpxyz(i,2),weight_mag_tmpxyz(i,3)
                        do j = 1, ntype
                            if (iatom_tmp2(i) == upfpsp(j)%num) weight_mag_tmpxyz(i,1:3) = weight_mag_tmpxyz(i,1:3) / upfpsp(j)%zp
                        end do
                        if(iatom_tmp2(i).ne.iatom_tmp(i)) then
                            write(6,*) "order of iatom in position/magnetic not the same", i
                            if(inode_tot .eq. 1) then
                                call mpi_abort(mpi_comm_world,ierr)
                            endif
                        endif
                    enddo
                endif


!********************************************************
                call scan_key_words (10, "LDAU_lambda", len("LDAU_lambda"), scanit)
                if(.not. scanit) then
                    do i=1,natom
                        LDAU_lambda_tmp(i,1)=0.0d0
                        LDAU_lambda_tmp(i,2)=0.0d0
                    enddo
                else
                    do i=1,natom
                        read(10, *) iatom_tmp2(i),LDAU_lambda_tmp(i,1),LDAU_lambda_tmp(i,2)
                    enddo
                    LDAU_lambda_tmp=LDAU_lambda_tmp/Hartree_eV
                endif
!********************************************************
                call scan_key_words (10, "Weight_atom", len("Weight_atom"), scanit)
                if(.not. scanit) then
                    do i=1,natom
                        weight_atom_tmp(i)=1.d0
                    enddo
                else
                    do i=1,natom
                        read(10, *) iatom_tmp2(i),weight_atom_tmp(i)
                    enddo
                endif
!********************************************************


                natom_coulomb=0
                natom_coulomb_fix=0
                call scan_key_words (10, "ENERGY_NATOM_COULOMB", len("ENERGY_NATOM_COULOMB"), scanit)
                if(.not. scanit) then
                natom_coulomb=0
                natom_coulomb_fix=0
                else
                    charge_natom_coulomb=0.d0
                  read(10,*) natom_coulomb
                    do i=1,natom_coulomb
                      read(10,*) ind_natom_coulomb(i),iatom_tmp2(i),alpha_natom_coulomb(1,i),alpha_natom_coulomb(2,i)
                      alpha_natom_coulomb(1,i)=(alpha_natom_coulomb(1,i)/0.529177d0)**2/4
                      alpha_natom_coulomb(2,i)=(alpha_natom_coulomb(2,i)/0.529177d0)**2/4
                    enddo
                  read(10,*) 
                  read(10,*) natom_coulomb_fix
                    do i=natom_coulomb+1,natom_coulomb+natom_coulomb_fix
                      read(10,*) ind_natom_coulomb(i),  &
                  iatom_tmp2(i),alpha_natom_coulomb(1,i),alpha_natom_coulomb(2,i), &
                   charge_natom_coulomb(1,i),charge_natom_coulomb(2,i)
                      alpha_natom_coulomb(1,i)=(alpha_natom_coulomb(1,i)/0.529177d0)**2/4
                      alpha_natom_coulomb(2,i)=(alpha_natom_coulomb(2,i)/0.529177d0)**2/4
                    enddo
                 endif




                close(10)
                !ccccccccccccccccccccccccccccccccccccccccccccccccccc
                !ccc Now, re-arrange xatom, so the same atoms are consequentive together. 
                !ccc This is useful to speed up the getwmask.f
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


                ii=0
                if (inode_tot.eq.1) then
                    open(9900, file = 'ORIGIN.INDEX')
                    rewind(9900)
                end if
                100    continue
                ncount=0
                itype_tmp=-2
                do i=1,natom
                    if(itype_tmp.eq.-2.and.iatom_tmp(i).ne.-1)itype_tmp=iatom_tmp(i)

                    if(iatom_tmp(i).eq.itype_tmp) then
                        ii=ii+1
                        ncount=ncount+1
                        iatom(ii)=iatom_tmp(i)
#ifdef TIMING
                        write ( 6, * ) "iatom(ii)", ii, iatom(ii)
#endif
                        iatom_tmp(i)=-1
                        xatom(:,ii)=xatom_tmp(:,i)
                        imov_at(:,ii)=imov_latt_vectmp(:,i)
                        VX_1(ii)=VX_1_tmp(i)
                        VY_1(ii)=VY_1_tmp(i)
                        VZ_1(ii)=VZ_1_tmp(i)
                        weight_mag(ii)=0.5d0+weight_mag_tmp(i)/2          ! weigh_mag: old meaning; input mag_tmp from - to +
                        weight_magxyz(ii,1)=weight_mag_tmpxyz(i,1)
                        weight_magxyz(ii,2)=weight_mag_tmpxyz(i,2)
                        weight_magxyz(ii,3)=weight_mag_tmpxyz(i,3)
                        langevin_factT(ii)=langevin_factT_tmp(i)
                        langevin_factG(ii)=langevin_factG_tmp(i)
                        constraint_mag_atom(ii)=constraint_mag_atom_tmp(i)
                        constraint_mag_alpha(ii)=constraint_mag_alpha_tmp(i)
                        LDAU_lambda(ii,:) = LDAU_lambda_tmp(i,:)
                        ind_order_new2old_tmp(ii)=i
                        ind_order_old2new_tmp(i)=ii
                        weight_atom_1(ii)=weight_atom_tmp(i)
                        if (inode_tot.eq.1) write(9900, '(i6,x,3i2,x,i6,x,i3)') i, imov_at(:,ii), ii, iatom(ii)
                    endif
                enddo
                if(ncount.gt.0) goto 100
                if(inode_tot.eq.1) close(9900)
                if(ii.ne.natom) then
                    write(message,*) "something wrong to rearrange xatom, stop"
                    call error_stop(message,__FILE__,__LINE__)
                endif

                do i=1,natom_coulomb+natom_coulomb_fix
                ind_natom_coulomb(i)=ind_order_old2new_tmp(ind_natom_coulomb(i))
                enddo

                       if(inode_tot.eq.1) then
                        open (unit = 2200, file = 'final.config')
                        rewind (2200)
                        write(2200, *) natom
                        write(2200, *)  "Lattice vector (Angstrom), stress(eV/natom)"
                        write(2200,"(3(E19.10,1x))") A_AU_1*AL(1,1),A_AU_1*AL(2,1),A_AU_1*AL(3,1)
                        write(2200,"(3(E19.10,1x))") A_AU_1*AL(1,2),A_AU_1*AL(2,2),A_AU_1*AL(3,2)
                        write(2200,"(3E19.10,1x))") A_AU_1*AL(1,3),A_AU_1*AL(2,3),A_AU_1*AL(3,3)
                        write(2200,*) "Position, move_x, move_y, move_z"
                        do ia = 1, natom
                            write(2200,1113) iatom(ia),xatom(1,ia),xatom(2,ia),xatom(3,ia), imov_at(1,ia),imov_at(2,ia),imov_at(3,ia)
                        enddo
                        close(2200)
                        1113   format(i4, 1x, 3(f14.9,1x),4x, 3(i1, 2x))
                       endif


                allocate(energy_coulomb(natom))
                deallocate(xatom_tmp)
                deallocate(imov_latt_vectmp)
                deallocate(iatom_tmp)
                deallocate(iatom_tmp2)
                deallocate(VX_1_tmp)
                deallocate(VY_1_tmp)
                deallocate(VZ_1_tmp)
                deallocate(weight_mag_tmp)
                deallocate(weight_mag_tmpxyz)
                deallocate(langevin_factT_tmp)
                deallocate(langevin_factG_tmp)
                deallocate(constraint_mag_atom_tmp)
                deallocate(constraint_mag_alpha_tmp)
                deallocate(ind_order_old2new_tmp)
                deallocate(ind_order_new2old_tmp)
                deallocate(LDAU_lambda_tmp)
                deallocate(weight_atom_tmp)


                return
        end subroutine readf_xatom_new

        !**************************************************
        subroutine split_xatom()
                implicit double precision (a-h,o-z)
                real*8, allocatable, dimension(:,:) :: dxatom

                allocate(dxatom(3,natom))

                do i=1,natom
                    dxatom(:,i)=xatom_all_1(:,i,Nimage+1)-xatom_all_1(:,i,0)
                enddo

                do i=1,natom
                    do j=1,3
                        if(dxatom(j,i).gt.0.5) dxatom(j,i)=dxatom(j,i)-1.d0
                        if(dxatom(j,i).lt.-0.5) dxatom(j,i)=dxatom(j,i)+1.d0
                    enddo
                enddo

                do image=1,Nimage
                    fac=image*1.d0/(Nimage+1)
                    xatom_all_1(:,:,image)=xatom_all_1(:,:,0)+dxatom(:,:)*fac
                enddo

                deallocate(dxatom)
                return
        end subroutine split_xatom
        subroutine split_xatom_inter(xbegin,xend,res)
                implicit none
                real*8, allocatable, dimension(:,:) :: dxatom
                real*8 :: xbegin(3,natom)
                real*8 :: xend(3,natom)
                real*8 :: res(3,natom,interNimage+1)
                real*8 :: fac
                integer i,j,image

                allocate(dxatom(3,natom))

                dxatom=xend-xbegin

                do i=1,natom
                    do j=1,3
                        if(dxatom(j,i).gt.0.5) dxatom(j,i)=dxatom(j,i)-1.d0
                        if(dxatom(j,i).lt.-0.5) dxatom(j,i)=dxatom(j,i)+1.d0
                    enddo
                enddo

                do image=1,interNimage
                    fac=image*1.d0/(interNimage+1)
                    res(:,:,image)=xbegin(:,:)+dxatom(:,:)*fac
                enddo
                res(:,:,interNimage+1)=xend(:,:)

                deallocate(dxatom)
                return
        end subroutine split_xatom_inter


        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        subroutine readf_xatom_multi(Nimage_tmp)
                implicit double precision (a-h,o-z)
                real*8, allocatable, dimension(:,:) :: xatom_tmp
                integer, allocatable, dimension(:,:) :: imov_latt_vectmp
                integer, allocatable, dimension(:)   ::  iatom_tmp

                open(10,file=f_xatom,status='old',action='read',iostat=ierr)

                if(ierr.ne.0) then
                    if(inode.eq.1) write(message,*) "IN.ATOM",f_xatom,"not exist, stop"
                    call error_stop(message,__FILE__,__LINE__) 
                endif

                rewind(10)

                do 10 image=0,Nimage_tmp+1
                    read(10,*) natom
                    if(natom.gt.matom_1) then
                        if(inode.eq.1) then
                            write(message,*) "natom.gt.matom_1, increase matom_1 in data.f, stop ",f_xatom,natom,matom_1
                        endif
                        call error_stop(message,__FILE__,__LINE__) 
                    endif

                    if(image.eq.0) then
                        allocate(xatom_tmp(3,natom))
                        allocate(imov_latt_vectmp(3,natom))
                        allocate(iatom_tmp(natom))
                    endif

                    read(10,*)     ! this is supposedly the lattice vector line
                    read(10,*) (AL(i,1),i=1,3)
                    read(10,*) (AL(i,2),i=1,3)
                    read(10,*) (AL(i,3),i=1,3)

                    AL = AL/A_AU_1 !zhilin

                    read(10,*)         ! this is supposedly the Position line
                    do i=1,natom
                        read(10,*) iatom_tmp(i),xatom_tmp(1,i),xatom_tmp(2,i),xatom_tmp(3,i),imov_latt_vectmp(1,i),imov_latt_vectmp(2,i),imov_latt_vectmp(3,i)
                    enddo
                    read(10,*)         ! this is supposedly for the force lines
                    do i=1,natom
                        read(10,*)         ! This is for the force lines
                    enddo

                    !ccccccccccccccccccccccccccccccccccccccccccccccccccc
                    !ccc Now, re-arrange xatom, so the same atoms are consequentive together. 
                    !ccc This is useful to speed up the getwmask.f
                    !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

                    ii=0
                    100    continue
                    ncount=0
                    itype_tmp=-2
                    do i=1,natom
                        if(itype_tmp.eq.-2.and.iatom_tmp(i).ne.-1)itype_tmp=iatom_tmp(i)

                        if(iatom_tmp(i).eq.itype_tmp) then
                            ii=ii+1
                            ncount=ncount+1
                            iatom(ii)=iatom_tmp(i)
                            iatom_tmp(i)=-1
                            xatom_all_1(:,ii,image)=xatom_tmp(:,i)
                            imov_at(:,ii)=imov_latt_vectmp(:,i)
                        endif
                    enddo
                    if(ncount.gt.0) goto 100
                    if(ii.ne.natom) then
                        write(message,*) "something wrong to rearrange xatom, stop"
                        call error_stop(message,__FILE__,__LINE__)
                    endif

                    read(10,*)       ! this is for the "-----------" line

                    10     continue    ! image

                    close(10)

                    deallocate(xatom_tmp)
                    deallocate(imov_latt_vectmp)
                    deallocate(iatom_tmp)

                    return
            end subroutine readf_xatom_multi
            function cross(a,b)
                real*8 :: a(3)
                real*8 :: b(3)
                real*8 :: cross
                real*8 :: tmp(3)
                tmp(1)=a(2)*b(3)-a(3)*b(2)
                tmp(2)=-a(1)*b(3)+a(3)*b(1)
                tmp(3)=a(1)*b(2)-a(2)*b(1)
                cross=dsqrt(dot_product(tmp,tmp))
            end function cross
            function itostr(i)
                integer,intent(in) :: i
                character(len=1024) :: itostr
                !
                write(itostr,*) i
                !
                itostr=adjustl(trim(itostr))
            end function itostr


    end subroutine input


