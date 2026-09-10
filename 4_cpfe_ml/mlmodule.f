	  module mlmodule
	  use, intrinsic :: iso_c_binding
!	  use wzzzinput, only: WZZZBAT_DIR, WZZZpycommand
	  implicit none
!
	  interface
      subroutine hello_world() bind (c)
      end subroutine hello_world
      end interface
!
	  contains
!
!
	  subroutine testsubroutine
	  implicit none
	  integer :: istat
	  integer :: i
	  real(8), dimension(5) :: input_data
!
!
	  input_data = [1.1d0, 2.2d0, 3.3d0, 4.4d0, 5.5d0]
	  open(500,file='../fortran_to_python.txt',action='write',
     + status='replace')
	  write(500, *) input_data
	  close(unit=500)
!	  
!	  
	  end subroutine testsubroutine
!	  
!
!
	  subroutine bat_cmd_build(bat_dir,bat_content)
	  implicit none
	  CHARACTER*200, INTENT(IN) :: bat_dir
	  CHARACTER*200, INTENT(IN) :: bat_content
!	  
	  open(unit=501, file=bat_dir,action='write',status='replace')
	  write(501, '(a)') '@echo off'//new_line('a')//trim(bat_content)
	  close(501)
!	  
	  end subroutine bat_cmd_build
!
	  subroutine bat_cmd_build2(bat_dir, line1, line2)
      implicit none
      CHARACTER*200, INTENT(IN) :: bat_dir
      CHARACTER*200, INTENT(IN) :: line1
      CHARACTER*200, INTENT(IN) :: line2
!
      open(unit=501, file=bat_dir, action='write', status='replace')
      write(501,'(a)') '@echo off'
      write(501,'(a)') trim(line1)
      write(501,'(a)') trim(line2)
      close(501)
      end subroutine bat_cmd_build2
!
	  subroutine bat_cmd_build3(bat_dir, line1, line2, line3)
      implicit none
      CHARACTER*200, INTENT(IN) :: bat_dir
      CHARACTER*200, INTENT(IN) :: line1
      CHARACTER*200, INTENT(IN) :: line2
      CHARACTER*200, INTENT(IN) :: line3
      open(unit=501, file=bat_dir, action='write', status='replace')
      write(501,'(a)') '@echo off'
      write(501,'(a)') trim(line1)
      write(501,'(a)') trim(line2)
      write(501,'(a)') trim(line3)
      close(501)
      end subroutine bat_cmd_build3
!
!
!
	  subroutine data_output(NTENS,NSTATV,NOEL,NPT)
	  implicit none
	  INTEGER, INTENT(IN) :: NTENS
	  INTEGER, INTENT(IN) :: NSTATV
	  INTEGER, INTENT(IN) :: NOEL
	  INTEGER, INTENT(IN) :: NPT
	  real(8), dimension(2) :: output_data
	  output_data = [NOEL,NPT]
	  open(502,file='../data_output.txt',action='write',
     + status='old', POSITION='APPEND')
	  write(502, *) output_data
	  close(unit=502)
	  end subroutine data_output
!
!
	  subroutine check_hangon_frag(frag_dir)
	  implicit none
	  LOGICAL EXISTS
	  CHARACTER*200, INTENT(IN) :: frag_dir
!	  
	  write(*,*) '**********check_hangon_frag:run***********'
 10   continue
	  INQUIRE(FILE=frag_dir, EXIST=EXISTS)  
      IF (.NOT. EXISTS) THEN  
         CALL SYSTEM('ping 127.0.0.1 -n 1 -w 1000 > nul')   
         GOTO 10
      END IF
	  write(*,*) '**********check_hangon_frag:done**********'  
!
	  end subroutine check_hangon_frag
!
!		1_prepare sym g
		subroutine get_symmetries(crystal_type, symmetries, n_sym)
			implicit none
			
			character(len=*), intent(in)  :: crystal_type
			double precision, intent(out) :: symmetries(3, 3, 24)
			integer, intent(out)          :: n_sym
			
			double precision :: m(3, 3), sz(3, 3), flip_x(3, 3)
			double precision :: angle, c, s, det, pi
			integer          :: p(6, 3), signs(8, 3)
			integer          :: i, j, k, ip, is, count
			
			symmetries = 0.0d0
			n_sym = 0
			pi = 3.1415926535897932d0

			if (trim(adjustl(crystal_type)) == "BCC") then
				p(1,:) = (/1, 2, 3/); p(2,:) = (/1, 3, 2/)
				p(3,:) = (/2, 1, 3/); p(4,:) = (/2, 3, 1/)
				p(5,:) = (/3, 1, 2/); p(6,:) = (/3, 2, 1/)
				
				count = 1
				do i = 1, -1, -2
					do j = 1, -1, -2
						do k = 1, -1, -2
							signs(count, :) = (/i, j, k/)
							count = count + 1
						end do
					end do
				end do
				
				count = 0
				do ip = 1, 6
					do is = 1, 8
						m = 0.0d0
						do i = 1, 3
							m(i, p(ip, i)) = dble(signs(is, i))
						end do
						
						det = m(1,1)*(m(2,2)*m(3,3) - m(2,3)*m(3,2)) - 
     +      m(1,2)*(m(2,1)*m(3,3) - m(2,3)*m(3,1)) + 
     +      m(1,3)*(m(2,1)*m(3,2) - m(2,2)*m(3,1))
						
						if (abs(det - 1.0d0) < 1.0d-6) then
							count = count + 1
							symmetries(:,:,count) = m
						end if
					end do
				end do
				n_sym = count

			else if (trim(adjustl(crystal_type)) == "HCP") then
				flip_x = 0.0d0
				flip_x(1,1) = 1.0d0; flip_x(2,2) = -1.0d0; flip_x(3,3) = -1.0d0
				
				count = 0
				do i = 0, 5
					angle = i * (pi / 3.0d0)
					c = cos(angle)
					s = sin(angle)
					
					sz = 0.0d0
					sz(1,1) = c;  sz(1,2) = -s; sz(1,3) = 0.0d0
					sz(2,1) = s;  sz(2,2) = c;  sz(2,3) = 0.0d0
					sz(3,1) = 0.0d0; sz(3,2) = 0.0d0; sz(3,3) = 1.0d0
					
					count = count + 1
					symmetries(:,:,count) = sz
					
					count = count + 1
					symmetries(:,:,count) = matmul(sz, flip_x)
				end do
				n_sym = count
			end if

		end subroutine get_symmetries
!		
!
		subroutine map_to_fr_vectorized(crystal_id, g_in, r_best, g_best)
			implicit none
			integer, intent(in)           :: crystal_id
			double precision, intent(in)  :: g_in(3, 3)
			double precision, intent(out) :: r_best(3)
			double precision, intent(out) :: g_best(3, 3)

			double precision :: syms(3, 3, 24), g_all(3, 3, 24),g_in_T(3, 3)
			double precision :: traces(24), cos_theta(24), theta(24)
			double precision :: n_all(3, 24), r_all(3, 24), norms(24)
			double precision :: scores(24), best_val, factor
			integer          :: n_sym, i, j, k, best_idx
			logical          :: in_fr(24)
			character(len=3) :: crystal_type
			
			double precision, parameter :: eps = 1.0d-7
			double precision, parameter :: pi  = 3.141592653589793d0
			
			if (crystal_id == 1) then
				crystal_type = "BCC"
			else if (crystal_id == 9) then
				crystal_type = "HCP"
			end if

			call get_symmetries(crystal_type, syms, n_sym)
!			g_in_T = transpose(g_in)

			do j = 1, 3
				do i = 1, 3
					g_all(i, j, 1:n_sym) = syms(i, 1, 1:n_sym) * g_in(1, j)
     +                         + syms(i, 2, 1:n_sym) * g_in(2, j)
     +                         + syms(i, 3, 1:n_sym) * g_in(3, j)
				end do
			end do

			traces(1:n_sym) = g_all(1, 1, :) + g_all(2, 2, :) + g_all(3, 3, :)
			cos_theta = (traces - 1.0d0) / 2.0d0
			
			where (cos_theta > 1.0d0)  cos_theta = 1.0d0
			where (cos_theta < -1.0d0) cos_theta = -1.0d0
			theta = acos(cos_theta)

			r_all = 0.0d0
			n_all(1, :) = g_all(3, 2, :) - g_all(2, 3, :)
			n_all(2, :) = g_all(1, 3, :) - g_all(3, 1, :)
			n_all(3, :) = g_all(2, 1, :) - g_all(1, 2, :)

			do k = 1, n_sym
				norms(k) = sqrt(n_all(1, k)**2 + n_all(2, k)**2 + n_all(3, k)**2)
				
				if (norms(k) > 1.0d-9 .and. theta(k) < (pi - 1.0d-6)) then
					factor = tan(theta(k) / 2.0d0) / norms(k)
					r_all(:, k) = n_all(:, k) * factor
				else if (theta(k) >= (pi - 1.0d-6)) then
					r_all(:, k) = 1.0d6
				end if
			end do

			in_fr = .false.
			if (trim(crystal_type) == "BCC") then
				do k = 1, n_sym
					if (all(abs(r_all(:, k)) <= 0.41421356d0 + eps) .and. 
     +          sum(abs(r_all(:, k))) <= 1.0d0 + eps) in_fr(k) = .true.
				end do
			else if (trim(crystal_type) == "HCP") then
				do k = 1, n_sym
					if (abs(r_all(3, k)) <= 0.26794919d0 + eps .and. 
     +            abs(r_all(2, k)) <= 0.57735027d0 + eps .and. 
     +            abs(0.8660254d0*r_all(1,k) + 0.5d0*r_all(2,k)) <= 
     +            0.57735027d0 + eps .and. 
     +            abs(0.8660254d0*r_all(1,k) - 0.5d0*r_all(2,k)) <= 
     +            0.57735027d0 + eps) in_fr(k) = .true.
				end do
			else
				in_fr(1:n_sym) = .true.
			end if

			do k = 1, n_sym
				scores(k) = sqrt(sum(r_all(:, k)**2))
				if (.not. in_fr(k)) scores(k) = scores(k) + 1.0d10
			end do

			best_idx = minloc(scores(1:n_sym), 1)

			r_best = r_all(:, best_idx)
			g_best = g_all(:, :, best_idx)

		end subroutine map_to_fr_vectorized
!
!
		subroutine mat_to_rodrigues_sub(R, r_vec)
			implicit none
			real(8), intent(in) :: R(3,3)
			real(8), intent(out) :: r_vec(3)
			real(8) :: tr
			
			tr = R(1,1) + R(2,2) + R(3,3)
			if (tr > -0.999999d0) then
				r_vec(1) = R(3,2) - R(2,3)
				r_vec(2) = R(1,3) - R(3,1)
				r_vec(3) = R(2,1) - R(1,2)
				r_vec = r_vec / (1.0d0 + tr)
			else
				r_vec = 0.0d0
			end if
		end subroutine
		
		subroutine is_in_fr_sub(r, phaid, flag)
			implicit none
			real(8), intent(in) :: r(3)
			integer, intent(in) :: phaid
			integer, intent(out) :: flag
			real(8) :: eps, lim, r1, r2, r3
			
			eps = 1.0d-7
			flag = 0
			
			if (phaid == 1) then

				if (abs(r(1)) > 0.41421356d0 + eps) return
				if (abs(r(2)) > 0.41421356d0 + eps) return
				if (abs(r(3)) > 0.41421356d0 + eps) return
				if ((abs(r(1)) + abs(r(2)) + abs(r(3))) > 1.0d0 + eps) return
				flag = 1
				
			else if (phaid == 9) then
				r1 = r(1); r2 = r(2); r3 = r(3)
				if (abs(r3) > 0.26794919d0 + eps) return
				lim = 0.57735027d0 + eps
				if (abs(r2) > lim) return
				if (abs(0.8660254d0 * r1 + 0.5d0 * r2) > lim) return
				if (abs(0.8660254d0 * r1 - 0.5d0 * r2) > lim) return
				flag = 1
			end if
		end subroutine
		
		subroutine rodrigues_to_mat_sub(r_in, R_mat)
			implicit none
			real(8), intent(in)  :: r_in(3)
			real(8), intent(out) :: R_mat(3,3)
			real(8) :: r2, rx, ry, rz
			
			! Fix: Use r_in to match the declaration 
			rx = r_in(1)
			ry = r_in(2)
			rz = r_in(3)
			r2 = rx*rx + ry*ry + rz*rz
			
			! Fix: Use R_mat to match the declaration 
			R_mat(1,1) = 1.0d0 + rx*rx - ry*ry - rz*rz
			R_mat(1,2) = 2.0d0 * (rx*ry - rz)
			R_mat(1,3) = 2.0d0 * (rx*rz + ry)
			
			R_mat(2,1) = 2.0d0 * (rx*ry + rz)
			R_mat(2,2) = 1.0d0 - rx*rx + ry*ry - rz*rz
			R_mat(2,3) = 2.0d0 * (ry*rz - rx)
			
			R_mat(3,1) = 2.0d0 * (rx*rz - ry)
			R_mat(3,2) = 2.0d0 * (ry*rz + rx)
			R_mat(3,3) = 1.0d0 - rx*rx - ry*ry + rz*rz
			
			R_mat = R_mat / (1.0d0 + r2)
		end subroutine
		
		subroutine get_crystal_symmetries(phaid, syms, nsym)
			  implicit none
			  integer, intent(in) :: phaid
			  real(8), intent(out) :: syms(24, 3, 3)
			  integer, intent(out) :: nsym
			  integer :: i, j, k, p, s, row, col
			  real(8) :: m(3,3), det, ang, c, s_ang
			  real(8) :: pi = 3.141592653589793d0
			  integer :: perms(6, 3), signs(8, 3)
			  real(8) :: flip_x(3, 3)
			  real(8) :: sz(3, 3)
			  nsym = 0
			  syms = 0.0d0
			  perms = reshape([1,2,3, 1,3,2, 2,1,3, 2,3,1, 
     +                 		   3,1,2, 3,2,1], [6,3])

			  signs = reshape([1,1,1, 1,1,-1, 1,-1,1, 1,-1,-1, 
     +                		  -1,1,1, -1,1,-1, -1,-1,1, -1,-1,-1], [8,3])

			  if (phaid == 1) then
				  do p = 1, 6
					  do s = 1, 8
						  m = 0.0d0
						  do row = 1, 3
							  m(row, perms(p, row)) = dble(signs(s, row))
						  end do
						  
						  det = m(1,1)*(m(2,2)*m(3,3) - m(2,3)*m(3,2))
     +                        - m(1,2)*(m(2,1)*m(3,3) - m(2,3)*m(3,1))
     +    					  + m(1,3)*(m(2,1)*m(3,2) - m(2,2)*m(3,1))
						  
						  if (abs(det - 1.0d0) < 1.0d-7) then
							  nsym = nsym + 1
							  syms(nsym, :, :) = m
						  end if
					  end do
				  end do
			  else if (phaid == 9) then
				  flip_x = 0.0d0
				  flip_x(1,1) = 1.0d0
				  flip_x(2,2) = -1.0d0
				  flip_x(3,3) = -1.0d0
				  
				  do i = 0, 5
					  ang = dble(i) * 60.0d0 * pi / 180.0d0
					  c = cos(ang)
					  s_ang = sin(ang)
					  
					  sz = 0.0d0
					  sz(1,1) = c
					  sz(1,2) = -s_ang
					  sz(1,3) = 0.0d0
					  sz(2,1) = s_ang
					  sz(2,2) = c
					  sz(2,3) = 0.0d0
					  sz(3,1) = 0.0d0
					  sz(3,2) = 0.0d0
					  sz(3,3) = 1.0d0
					  
					  nsym = nsym + 1
					  syms(nsym, :, :) = sz
					  
					  nsym = nsym + 1
					  syms(nsym, :, :) = matmul(sz, flip_x)
				  end do
			  end if
			  end subroutine get_crystal_symmetries
		
		subroutine map_to_rodrigues_fr(phaid, R_in, R_out)
			use utilities, only : mmult
			implicit none
			integer, intent(in) :: phaid
			real(8), intent(in) :: R_in(3,3)
			real(8), intent(out) :: R_out(3,3)
			
			real(8) :: syms(24,3,3), G_equiv(3,3), r_cand(3), best_r(3)
			real(8) :: min_norm, current_norm
			integer :: nsym, i, in_fr_flag, found_in_fr
			
			call get_crystal_symmetries(phaid, syms, nsym)
			
			found_in_fr = 0
			min_norm = 1.0d10
			best_r = 0.0d0
			
			do i = 1, nsym
				call mmult(syms(i,:,:), R_in, G_equiv)
				call mat_to_rodrigues_sub(G_equiv, r_cand)
				
				call is_in_fr_sub(r_cand, phaid, in_fr_flag)
				
				if (in_fr_flag == 1) then
					current_norm = sqrt(r_cand(1)**2 + r_cand(2)**2 + r_cand(3)**2)
					if (current_norm < min_norm) then
						min_norm = current_norm
						best_r = r_cand
						found_in_fr = 1
					end if
				end if
			end do
			
			if (found_in_fr == 0) then
				min_norm = 1.0d10
				do i = 1, nsym
					call mmult(syms(i,:,:), R_in, G_equiv)
					call mat_to_rodrigues_sub(G_equiv, r_cand)
					current_norm = sqrt(r_cand(1)**2 + r_cand(2)**2 + r_cand(3)**2)
					if (current_norm < min_norm) then
						min_norm = current_norm
						best_r = r_cand
					end if
				end do
			end if
			
			call rodrigues_to_mat_sub(best_r, R_out)
			
		end subroutine
		
!     ===========================================================
!     LOAD ALL LSTM SYSTEMS (FIXED FORM)
!     ===========================================================
      subroutine LOAD_ALL_LSTM_SYSTEMS()
      use globalvariables
      use wzzzinput
      implicit none
      
      write(*,*) '--- [MLMODULE] LOADING LSTM SYSTEMS (LOP=0) ---'
      
      call READ_SINGLE_LSTM(Wzinput_lstm_bcc_path, 1)
      call READ_SINGLE_LSTM(Wzinput_lstm_hcp_path, 2)

      MAX_LSTM_HIDDEN_SIZE = max(BCC_N_hidden, HCP_N_hidden)
      
      write(*,*) '--- ALLOCATING LSTM STATES ---'
      write(*,*) '    Elements:', ELEMENT_NUM
      write(*,*) '    IPs/Elem:', ELEMENT_NODE
      write(*,*) '    Max Hidden Size:', MAX_LSTM_HIDDEN_SIZE
      
      
      write(*,*) '--- LSTM INITIALIZATION COMPLETE ---'
      
      end subroutine LOAD_ALL_LSTM_SYSTEMS

!     ===========================================================
!     INTERNAL HELPER
!     ===========================================================
	  subroutine READ_SINGLE_LSTM(fpath, mode)
      use globalvariables
	  use wzzzinput, only : BCC_LSTM_N_IN,
     + BCC_LSTM_N_HIDDEN,BCC_LSTM_N_LAYERS,
     + BCC_LSTM_N_OUT,HCP_LSTM_N_IN,
     + HCP_LSTM_N_HIDDEN,HCP_LSTM_N_LAYERS,
     + HCP_LSTM_N_OUT
      implicit none
      character(len=*), intent(in) :: fpath
      integer, intent(in) :: mode
      integer :: fid=108, ilay, cur_in, h_sz, l_sz
	  integer :: N_in, N_hid, N_lay, N_out
	  real(8), allocatable :: Temp_W_ih(:,:), Temp_W_hh(:,:), Temp_B(:), Temp_W_fc(:,:)
	  
!	  real(8) :: Test_W_Fortran(3, 2)
!     real(8) :: W_Final(2, 3)
!     open(unit=888, file='F:/test_weight.txt', status='old', action='read')
!      read(888, *) Test_W_Fortran
!      close(888)
!      W_Final = transpose(Test_W_Fortran)
!      write(*,*) 'Raw Read (Column 1):', Test_W_Fortran(:, 1)
!      write(*,*) 'Raw Read (Column 2):', Test_W_Fortran(:, 2)
!     write(*,*) 'Transposed (Row 1):', W_Final(1, :)


      open(unit=fid, file=fpath, status='old', action='read')
      read(fid, *) N_in, N_hid, N_lay, N_out

      if (mode == 1) then
          if (N_in /= BCC_LSTM_N_IN .or.
     +        N_hid /= BCC_LSTM_N_HIDDEN .or.
     +        N_lay /= BCC_LSTM_N_LAYERS .or.
     +        N_out /= BCC_LSTM_N_OUT) then

              write(*,*) 'ERROR: BCC LSTM size mismatch'
              write(*,*) 'FILE:',N_in,N_hid,N_lay,N_out
              write(*,*) 'CFG :',BCC_LSTM_N_IN,
     +                   BCC_LSTM_N_HIDDEN,
     +                   BCC_LSTM_N_LAYERS,
     +                   BCC_LSTM_N_OUT
              stop
          endif

          BCC_N_in = BCC_LSTM_N_IN
          BCC_N_hidden = BCC_LSTM_N_HIDDEN
          BCC_N_layers = BCC_LSTM_N_LAYERS
          BCC_N_out = BCC_LSTM_N_OUT
          allocate(BCC_W_ih(N_lay, 4*N_hid, max(N_in, N_hid)))
          allocate(BCC_W_hh(N_lay, 4*N_hid, N_hid))
          allocate(BCC_B_bias(N_lay, 4*N_hid))
          allocate(BCC_In_Mean(N_in), BCC_In_Std(N_in), BCC_Out_Mean(N_out), BCC_Out_Std(N_out))
          read(fid, *) BCC_In_Mean, BCC_In_Std, BCC_Out_Mean, BCC_Out_Std
        
          do ilay = 1, N_lay
              cur_in = merge(N_in, N_hid, ilay == 1)
              allocate(Temp_W_ih(4*N_hid, cur_in)) 
			  read(fid, *) Temp_W_ih
			  BCC_W_ih(ilay, 1:4*N_hid, 1:cur_in) = Temp_W_ih
              deallocate(Temp_W_ih)

              allocate(Temp_W_hh(4*N_hid, N_hid))
			  read(fid, *) Temp_W_hh
			  BCC_W_hh(ilay, :, :) = Temp_W_hh
              deallocate(Temp_W_hh)

              read(fid, *) BCC_B_bias(ilay, :)
              allocate(Temp_B(4*N_hid)); read(fid, *) Temp_B
              BCC_B_bias(ilay, :) = BCC_B_bias(ilay, :) + Temp_B
              deallocate(Temp_B)
          end do
          allocate(BCC_W_fc(N_out, N_hid), BCC_B_fc(N_out))
          allocate(Temp_W_fc(N_out, N_hid))
		  read(fid, *) Temp_W_fc
          BCC_W_fc = Temp_W_fc
		  read(fid, *) BCC_B_fc
          deallocate(Temp_W_fc)

      else if (mode == 2) then
          if (N_in /= HCP_LSTM_N_IN .or.
     +        N_hid /= HCP_LSTM_N_HIDDEN .or.
     +        N_lay /= HCP_LSTM_N_LAYERS .or.
     +        N_out /= HCP_LSTM_N_OUT) then

              write(*,*) 'ERROR: HCP LSTM size mismatch'
              write(*,*) 'FILE:',N_in,N_hid,N_lay,N_out
              write(*,*) 'CFG :',HCP_LSTM_N_IN,
     +                   HCP_LSTM_N_HIDDEN,
     +                   HCP_LSTM_N_LAYERS,
     +                   HCP_LSTM_N_OUT
              stop
          endif

          HCP_N_in = HCP_LSTM_N_IN
          HCP_N_hidden = HCP_LSTM_N_HIDDEN
          HCP_N_layers = HCP_LSTM_N_LAYERS
          HCP_N_out = HCP_LSTM_N_OUT
          allocate(HCP_W_ih(N_lay, 4*N_hid, max(N_in, N_hid)))
          allocate(HCP_W_hh(N_lay, 4*N_hid, N_hid))
          allocate(HCP_B_bias(N_lay, 4*N_hid))
          allocate(HCP_In_Mean(N_in), HCP_In_Std(N_in), HCP_Out_Mean(N_out), HCP_Out_Std(N_out))
          read(fid, *) HCP_In_Mean, HCP_In_Std, HCP_Out_Mean, HCP_Out_Std
        
          do ilay = 1, N_lay
              cur_in = merge(N_in, N_hid, ilay == 1)
              allocate(Temp_W_ih(4*N_hid, cur_in))
              read(fid, *) Temp_W_ih
              HCP_W_ih(ilay, 1:4*N_hid, 1:cur_in) = Temp_W_ih 
              deallocate(Temp_W_ih)

              allocate(Temp_W_hh(4*N_hid, N_hid))            
              read(fid, *) Temp_W_hh
              HCP_W_hh(ilay, :, :) = Temp_W_hh               
              deallocate(Temp_W_hh)

              read(fid, *) HCP_B_bias(ilay, :)
              allocate(Temp_B(4*N_hid)); read(fid, *) Temp_B
              HCP_B_bias(ilay, :) = HCP_B_bias(ilay, :) + Temp_B
              deallocate(Temp_B)
          end do
          allocate(HCP_W_fc(N_out, N_hid), HCP_B_fc(N_out))
          allocate(Temp_W_fc(N_out, N_hid))                  
          read(fid, *) Temp_W_fc
          HCP_W_fc = Temp_W_fc
          read(fid, *) HCP_B_fc
          deallocate(Temp_W_fc)
      end if
      close(fid)
	  end subroutine
!
!     ===========================================================
!     LSTM sigmoid
!     ===========================================================
	  real(8) function sigmoid(x)
      implicit none
      real(8), intent(in) :: x
      sigmoid = 1.0d0 / (1.0d0 + exp(-x))
      end function sigmoid
!
!     ===========================================================
!     LSTM FORWARD PASS (INFERENCE)
!     ===========================================================
      subroutine LSTM_FORWARD(noel, npt, phase_id, input_vec,
     +                        gmatinv, output_vec)

      use globalvariables
      use wzzzinput, only : LSTM_CFG_MAX_HIDDEN,
     + LSTM_CFG_MAX_LAYERS
      use utilities, only : vecmat6, matvec6
      implicit none

      integer, intent(in) :: noel, npt, phase_id
      real(8), intent(in) :: input_vec(:)
      real(8), intent(in) :: gmatinv(3,3)
      real(8), intent(out) :: output_vec(:)

      real(8) :: R_std(3,3)

      integer :: N_in, N_hid, N_out, N_lay
      integer :: ilay, i, in_dim

!     Hidden/cell workspace:
!     Fixed upper limits are defined manually in wzzzinput.f
!     BCC and HCP can have different hidden sizes / layer numbers.
      real(8) :: h_state(LSTM_CFG_MAX_HIDDEN,
     +                   LSTM_CFG_MAX_LAYERS)

      real(8) :: c_state(LSTM_CFG_MAX_HIDDEN,
     +                   LSTM_CFG_MAX_LAYERS)

!     Restore old-version local allocatable workspace
      real(8), allocatable :: curr_input(:)
      real(8), allocatable :: layer_output(:)
      real(8), allocatable :: gates_raw(:)

      real(8) :: i_gate, f_gate, g_gate, o_gate
      real(8) :: c_prev_val, h_prev_val
      real(8) :: c_curr_val, h_curr_val

      real(8) :: sig_cry_33(3,3)
      real(8) :: sig_sam_33(3,3)

      real(8) :: sigmoid


!     -----------------------------------------------------------
!     Sigmoid
!     -----------------------------------------------------------
      sigmoid(h_prev_val) =
     +    1.0d0 / (1.0d0 + exp(-h_prev_val))


!     -----------------------------------------------------------
!     Initialize local hidden/cell workspace
!     -----------------------------------------------------------
      h_state = 0.0d0
      c_state = 0.0d0


!     -----------------------------------------------------------
!     1. Get network dimensions and confirmed states
!     -----------------------------------------------------------
      if (phase_id == 1) then

!         BCC
          N_in  = BCC_N_in
          N_hid = BCC_N_hidden
          N_lay = BCC_N_layers
          N_out = BCC_N_out

          h_state(1:N_hid,1:N_lay) =
     +        BCC_h_Confirmed(noel,npt,1:N_hid,1:N_lay)

          c_state(1:N_hid,1:N_lay) =
     +        BCC_c_Confirmed(noel,npt,1:N_hid,1:N_lay)


      elseif (phase_id == 9) then

!         HCP
          N_in  = HCP_N_in
          N_hid = HCP_N_hidden
          N_lay = HCP_N_layers
          N_out = HCP_N_out

          h_state(1:N_hid,1:N_lay) =
     +        HCP_h_Confirmed(noel,npt,1:N_hid,1:N_lay)

          c_state(1:N_hid,1:N_lay) =
     +        HCP_c_Confirmed(noel,npt,1:N_hid,1:N_lay)


      else

          return

      endif


!     -----------------------------------------------------------
!     2. Input normalization
!     Restore old-version allocation logic
!     -----------------------------------------------------------
      allocate(curr_input(N_in))

      if (phase_id == 1) then

          curr_input =
     +        (input_vec(1:N_in)-BCC_In_Mean)
     +        / BCC_In_Std

      else

          curr_input =
     +        (input_vec(1:N_in)-HCP_In_Mean)
     +        / HCP_In_Std

      endif


!     -----------------------------------------------------------
!     3. LSTM layers
!     -----------------------------------------------------------
      do ilay = 1, N_lay

!         Allocate workspace for current phase/current layer
          allocate(layer_output(N_hid))
          allocate(gates_raw(4*N_hid))

          in_dim = size(curr_input)


!         -------------------------------------------------------
!         Gates
!         -------------------------------------------------------
          if (phase_id == 1) then

              gates_raw =
     +          matmul(
     +          BCC_W_ih(ilay,:,1:in_dim),
     +          curr_input)
     +          +
     +          matmul(
     +          BCC_W_hh(ilay,:,:),
     +          h_state(1:N_hid,ilay))
     +          +
     +          BCC_B_bias(ilay,:)

          else

              gates_raw =
     +          matmul(
     +          HCP_W_ih(ilay,:,1:in_dim),
     +          curr_input)
     +          +
     +          matmul(
     +          HCP_W_hh(ilay,:,:),
     +          h_state(1:N_hid,ilay))
     +          +
     +          HCP_B_bias(ilay,:)

          endif


!         -------------------------------------------------------
!         LSTM cell update
!         -------------------------------------------------------
          do i = 1, N_hid

              i_gate = sigmoid(gates_raw(i))

              f_gate =
     +            sigmoid(gates_raw(N_hid+i))

              g_gate =
     +            tanh(gates_raw(2*N_hid+i))

              o_gate =
     +            sigmoid(gates_raw(3*N_hid+i))


              c_prev_val =
     +            c_state(i,ilay)

              c_curr_val =
     +            f_gate*c_prev_val
     +            + i_gate*g_gate

              h_curr_val =
     +            o_gate*tanh(c_curr_val)


              layer_output(i) = h_curr_val

              h_state(i,ilay) = h_curr_val
              c_state(i,ilay) = c_curr_val

          enddo


!         -------------------------------------------------------
!         Output of this layer -> input of next layer
!
!         Restore old-version allocate/deallocate logic
!         -------------------------------------------------------
          deallocate(curr_input)

          allocate(curr_input(N_hid))

          curr_input = layer_output

          deallocate(layer_output)
          deallocate(gates_raw)

      enddo


!     -----------------------------------------------------------
!     4. Fully connected output + de-normalization
!     -----------------------------------------------------------
      if (phase_id == 1) then

          output_vec =
     +        matmul(BCC_W_fc,curr_input)
     +        + BCC_B_fc

          output_vec =
     +        output_vec*BCC_Out_Std
     +        + BCC_Out_Mean

      else

          output_vec =
     +        matmul(HCP_W_fc,curr_input)
     +        + HCP_B_fc

          output_vec =
     +        output_vec*HCP_Out_Std
     +        + HCP_Out_Mean

      endif


!     -----------------------------------------------------------
!     5. Crystal -> sample coordinate transformation
!     -----------------------------------------------------------
      R_std(1,1) = input_vec(14)
      R_std(1,2) = input_vec(15)
      R_std(1,3) = input_vec(16)

      R_std(2,1) = input_vec(17)
      R_std(2,2) = input_vec(18)
      R_std(2,3) = input_vec(19)

      R_std(3,1) = input_vec(20)
      R_std(3,2) = input_vec(21)
      R_std(3,3) = input_vec(22)


      call vecmat6(output_vec,sig_cry_33)

      sig_sam_33 =
     +    matmul(
     +    R_std,
     +    matmul(sig_cry_33,transpose(R_std)))

      call matvec6(sig_sam_33,output_vec)


!     -----------------------------------------------------------
!     6. Save current hidden/cell states
!     -----------------------------------------------------------
      if (phase_id == 1) then

          BCC_h_Current(noel,npt,1:N_hid,1:N_lay) =
     +        h_state(1:N_hid,1:N_lay)

          BCC_c_Current(noel,npt,1:N_hid,1:N_lay) =
     +        c_state(1:N_hid,1:N_lay)

      else

          HCP_h_Current(noel,npt,1:N_hid,1:N_lay) =
     +        h_state(1:N_hid,1:N_lay)

          HCP_c_Current(noel,npt,1:N_hid,1:N_lay) =
     +        c_state(1:N_hid,1:N_lay)

      endif


!     -----------------------------------------------------------
!     7. Release local workspace
!     -----------------------------------------------------------
      if (allocated(curr_input)) then
          deallocate(curr_input)
      endif


      end subroutine LSTM_FORWARD
!
!     ===========================================================
!     LSTM CALL
!     ===========================================================
	  subroutine PREDICT_STRESS_PACKED(noel, npt, phase_id, packed_input, gmatinv, sigma_out)
		  use globalvariables
		  implicit none
		  
		  integer, intent(in) :: noel, npt, phase_id
		  real(8), intent(in) :: packed_input(22) 
		  real(8), intent(in) :: gmatinv(3,3)
		  real(8), intent(out) :: sigma_out(6)

		  call LSTM_FORWARD(noel, npt, phase_id, packed_input, gmatinv, sigma_out)

      end subroutine PREDICT_STRESS_PACKED

	end module mlmodule