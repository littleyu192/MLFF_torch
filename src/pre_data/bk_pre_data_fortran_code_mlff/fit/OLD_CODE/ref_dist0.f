       program add_reference
       implicit double precision (a-h,o-z)

       real*8,allocatable,dimension(:,:) :: feat_ref,feat_case
       real*8,allocatable,dimension(:) :: Ei_case,Ei_ref
       real*8,allocatable,dimension(:,:) :: Gfeat_case
       integer,allocatable,dimension(:) :: ind_ref
       real*8,allocatable,dimension(:) :: weight_ref
       real*8,allocatable,dimension(:) :: w_feat
       real*8,allocatable,dimension(:,:) :: S


       real*8,allocatable,dimension(:) :: work,BB
       real*8,allocatable,dimension(:) :: E_fit
       real*8,allocatable,dimension(:) :: dist_min,dist0_ref
     
       integer,allocatable,dimension(:) :: ipiv
       integer,allocatable,dimension(:) :: iref_min,ncase_ref
       integer lwork


       open(10,file="feat_new_stored.1",form="unformatted")
       rewind(10)
       read(10) num_case,nfeat
       write(6,*) "num_case,nfeat", num_case,nfeat
       allocate(Ei_case(num_case))
       allocate(feat_case(nfeat,num_case))
       do ii=1,num_case
       read(10) jj,Ei_case(ii),feat_case(:,ii)
       enddo
       close(10)

       allocate(w_feat(nfeat))
       open(10,file="weight_feat.1")
       rewind(10)
       do j=1,nfeat
       read(10,*) j1,w_feat(j)
       w_feat(j)=w_feat(j)**2
       enddo
       close(10)



       open(12,file="Ind_reference.1")
       rewind(12)
       read(12,*) num_ref
       num_ref_tot=2*num_ref

       write(6,*) "num_ref points=", num_ref
       allocate(ind_ref(num_ref_tot))
       allocate(weight_ref(num_ref_tot))
       do i=1,num_ref
       read(12,*) i1,ind_ref(i),weight_ref(i)
       enddo
       close(12)


       allocate(feat_ref(nfeat,num_ref_tot))
       do i=1,num_ref
       do j=1,nfeat
       feat_ref(j,i)=feat_case(j,ind_ref(i))
       enddo
       enddo
cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       write(6,*) "recalculate Gfeat"

!       allocate(Gfeat_case(num_ref,num_case))
       allocate(dist_min(num_case))
       allocate(iref_min(num_case))
       do ii=1,num_case
       dist_min_tmp=100000.d0
       iref_tmp=0
       do ii2=1,num_ref
       dist=0.d0
       do j=1,nfeat-1     ! The last feature one is 1. 
       dist=dist+(feat_case(j,ii)-feat_ref(j,ii2))**2*
     &   w_feat(j)
       enddo
       if(dist.lt.dist_min_tmp) then
       dist_min_tmp=dist
       iref_tmp=ii2
       endif
       enddo
       dist_min(ii)=dist_min_tmp
       iref_min(ii)=iref_tmp
       enddo

ccccccccccccccccccccccccccccccccccccc
       allocate(dist0_ref(num_ref))
       allocate(ncase_ref(num_ref))

       do ii2=1,num_ref
   
       num=0
       ave_dist=0.d0
       do ii=1,num_case
       if(iref_min(ii).eq.ii2.and.dist_min(ii).gt.1.E-5) then
       ave_dist=ave_dist+dist_min(ii)
       num=num+1
       endif
       enddo
       
       if(num.gt.0) then
       dist0_ref(ii2)=ave_dist/num
       ncase_ref(ii2)=num
       else
       dist0_ref(ii2)=2.d0
       ncase_ref(ii2)=num
       endif

       enddo
ccccccccccccccccccccccccccccccccccccccccc

       open(12,file="Ind_reference.1.rev")
       rewind(12)
       write(12,*) num_ref
       do i=1,num_ref
       write(12,"(2(i6,1x),2(E15.6,2x),i6)") i1,ind_ref(i),
     &   weight_ref(i), dist0_ref(i),ncase_ref(i)
       enddo
       close(12)


       stop
       end




       
     



       
