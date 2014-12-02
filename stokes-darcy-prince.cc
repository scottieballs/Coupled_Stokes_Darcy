/* ---------------------------------------------------------------------
 * $Id: step-46.cc 30526 2013-08-29 20:06:27Z felix.gruber $
 *
 * Copyright (C) 2011 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2011
 * stokes-darcy.cc - modified: Prince Chidyagwai, Scott Ladenheim 2014
 */


// @sect3{Include files}

// The include files for this program are the same as for many others
// before. The only new one is the one that declares FE_Nothing as discussed
// in the introduction. The ones in the hp directory have already been
// discussed in step-27.

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/iterative_inverse.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <ostream>
#include <iterator>
#include <vector>
#include <sstream>


namespace CoupledProblem
{
using namespace dealii;

// @sect3{The <code>StokesDarcy</code> class template}

// This is the main class. It is, if you want, a combination of step-8 and
// step-22 in that it has member variables that either address the global
// problem (the Triangulation and hp::DoFHandler objects, as well as the
// hp::FECollection and various linear algebra objects) or that pertain to
// either the darcy or Stokes sub-problems. The general structure of
// the class, however, is like that of most of the other programs
// implementing stationary problems.
//
// There are a few helper functions (<code>cell_is_in_fluid_domain,
// cell_is_in_darcy_domain</code>) of self-explanatory nature (operating on
// the symbolic names for the two subdomains that will be used as
// material_ids for cells belonging to the subdomains, as explained in the
// introduction) and a few functions (<code>make_grid,
// set_active_fe_indices, assemble_interface_terms</code>) that have been
// broken out of other functions that can be found in many of the other
// tutorial programs and that will be discussed as we get to their
// implementation.
//
// The final set of variables (<code>viscosity, lambda, eta</code>)
// describes the material properties used for the two physics models.
template <int dim>
class StokesDarcy
{
public:
	StokesDarcy (const unsigned int stokes_degree, const unsigned int darcy_degree,
			const unsigned int pre_type);
	void run ();

private:
	enum
	{
		fluid_domain_id,
		darcy_domain_id
	};
	static bool
	cell_is_in_fluid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);
	static bool
	cell_is_in_fluid_domain_mg (const typename DoFHandler<dim>::cell_iterator &cell);
	static bool
	cell_is_in_darcy_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);
	static bool
	cell_is_in_darcy_domain_mg (const typename DoFHandler<dim>::cell_iterator &cell);

	void make_grid ();
	void set_active_fe_indices ();
	void setup_dofs (const unsigned int block_pattern);
	void setup_darcy_mg_dofs ();
	void setup_stokes_mg_dofs();
	void assemble_system ();

	void assemble_interface_term (const FEFaceValuesBase<dim>          &darcy_fe_face_values,             //Darcy values on interface
			const FEFaceValuesBase<dim>          &stokes_fe_face_values,            //Stokes values on interface
			std::vector<double>                  &darcy_phi_p,                      //Darcy pressure basis functions
			std::vector<SymmetricTensor<2,dim> > &stokes_symgrad_phi_u,             //stokes symmetric gradient
			std::vector<Tensor<1,dim> >          &stokes_phi_u,
			std::vector<double>                  &stokes_phi_p,                     //stokes velocity basis functions
			FullMatrix<double>                   &local_interface_matrix_velocity_pressure, //
			FullMatrix<double>                   &local_interface_matrix_pressure_velocity,
			FullMatrix<double>                   &local_interface_matrix_velocity_velocity) const;    //all terms on the interface

	void assemble_darcy_multigrid();
	void assemble_stokes_multigrid();

	void construct_darcy_preconditioner();
	void construct_stokes_preconditioner();

	void solve (const unsigned int refinement_cycle, const unsigned int solution_type, const unsigned int block_pattern) ;
	void output_results (const unsigned int refinement_cycle) const;
	void refine_mesh ();
	void compute_errors(const unsigned int refinement_cycle) ;

	//degrees of freedom
	const unsigned int    stokes_degree;
	const unsigned int    darcy_degree;

	Triangulation<dim>    triangulation;

	FESystem<dim>         stokes_fe;              //stokes system
	FESystem<dim>         darcy_fe;               //darcy system
	FESystem<dim> 		  stokes_mg_fe;
	hp::FECollection<dim> fe_collection;
	hp::DoFHandler<dim>   dof_handler;

	ConstraintMatrix      constraints;

	BlockSparsityPattern       sparsity_pattern;
	BlockSparseMatrix<double>  system_matrix;

	BlockVector<double>        solution;
	BlockVector<double>        system_rhs;

	const double        viscosity;
	const double        mu;
	const double        G;
	const double        None;
	const unsigned int	pre_type;

	ConvergenceTable     convergence_table;
	ConvergenceTable	 iteration_table;

	//Darcy Multigrid Objects
	Triangulation<dim> 	darcy_tri;
	DoFHandler<dim>   	darcy_dof_handler;
	ConstraintMatrix 	mg_constraints_darcy;
	SparsityPattern		sparsity_pattern_darcy;

	MGLevelObject<SparsityPattern>       darcy_mg_sparsity_patterns;
	MGLevelObject<SparseMatrix<double> > darcy_mg_matrices;
	MGLevelObject<SparseMatrix<double> > darcy_mg_interface_matrices;
	MGConstrainedDoFs                    darcy_mg_constrained_dofs;

	//Stokes Multigrid Objects
	Triangulation<dim> stokes_tri;
	DoFHandler<dim>   stokes_dof_handler;
	ConstraintMatrix mg_constraints_stokes;
	SparsityPattern sparsity_pattern_stokes;

	MGLevelObject<SparsityPattern>       stokes_mg_sparsity_patterns;
	MGLevelObject<SparseMatrix<double> > stokes_mg_matrices;
	MGLevelObject<SparseMatrix<double> > stokes_mg_interface_matrices;
	MGConstrainedDoFs                    stokes_mg_constrained_dofs;

};

//Stokes velocity boundary values
template <int dim>
class StokesBoundaryValues : public Function<dim>
{
public:
	StokesBoundaryValues () : Function<dim>(dim+1+1) {}

	virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;

	virtual void vector_value (const Point<dim> &p,
			Vector<double>   &value) const;
};


template <int dim>
double
StokesBoundaryValues<dim>::value (const Point<dim>  &p,
		const unsigned int component) const
		{
	Assert (component < this->n_components,
			ExcIndexRange (component, 0, this->n_components));


	if(component == 0)
	{
		switch(dim)
		{
		case 2:
		{
			return p[1]*p[1] -2.0*p[1] + 2.0*p[1] + 2.0*p[0]; //0.0
		}
		case 3:
		{
			return 0.0;

		}
		default:
			Assert (false, ExcNotImplemented());
		}
	}
	else if (component == dim-1)
	{
		switch (dim)
		{
		case 2:
		{
			return p[0]*p[0] - p[0] -2.0*p[1]; //-1.0;
		}
		case 3:
		{
			return -1.0;
		}
		default:
			Assert (false, ExcNotImplemented());
		}
	}
	else
	{
		return 0;
	}

		}

template <int dim>
void
StokesBoundaryValues<dim>::vector_value (const Point<dim> &p,
		Vector<double>   &values) const
		{
	for (unsigned int c=0; c<this->n_components; ++c)
		values(c) = StokesBoundaryValues<dim>::value (p, c);
		}


//Darcy pressure boundary values
template <int dim>
class DarcyBoundaryValues : public Function<dim>
{
public:
	DarcyBoundaryValues () : Function<dim>(dim+1+1) {}

	virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;


};


template <int dim>
double
DarcyBoundaryValues<dim>::value (const Point<dim>  &p,
		const unsigned int /*component*/) const
		{
	switch (dim)
	{
	case 2:
		return -1.0*pow(p[0],2.0)*p[1] +p[0]*p[1] + std::pow(p[1],2.0);//p[1]-1.0;
	case 3:
		return p[2]-1.0;
	default:
		Assert (false, ExcNotImplemented());
	}

	return 0;
		}

//Darcy right hand side

template <int dim>
class DarcyRightHandSide: public Function<dim>
{
public:
	DarcyRightHandSide() : Function<dim>(){}

	virtual double value(const Point<dim> &p,
			const unsigned int component =0) const;
};

template<int dim>
double DarcyRightHandSide<dim>::value(const Point<dim> &p,
		const unsigned int /*component*/)const
		{

	switch(dim)
	{
	case 2:
		return 2.0*p[1]-2.0; //0.0;
	case 3:
		return 0.0;
	default:
		Assert(false,ExcNotImplemented());
	}
		}


//Stokes right hand side
template <int dim>
class RightHandSide : public Function<dim>
{
public:
	RightHandSide () : Function<dim>(dim+1) {}


	virtual void vector_value (const Point<dim>   &p,
			Vector<double> &values) const;

	virtual void vector_value_list (const std::vector<Point<dim> > &points,
			std::vector <Vector<double>>   &value_list) const;

};


template <int dim>
inline
void RightHandSide<dim>::vector_value (const Point<dim> &p,
		Vector<double>   &values) const
		{
	//Assert (values.size() == dim+1,
	//        ExcDimensionMismatch (values.size(), dim));
	//Assert (dim >= 2, ExcNotImplemented());


	switch (dim)
	{
	case 2:
	{
		values(0) = -1.0*2.0-2.0*p[0]*p[1] + p[1]; //0.0;
		values(1) = -1.0*2.0-p[0]*p[0] + p[0] + 2.0*p[1];//1.0;
		return;
	}
	case 3:
	{
		values(0) = 0.0;
		values(1) = 0.0;
		values(2) = 1.0;
		break;
	}
	default:
		Assert (false, ExcNotImplemented());
	}
		}//RightHandside<dim>::vector_value

template <int dim>
void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points,
		std::vector<Vector<double> >   &value_list) const
		{
	//Assert (value_list.size() == points.size(),
	//        ExcDimensionMismatch (value_list.size(), points.size()));

	const unsigned int n_points = points.size();

	for (unsigned int p=0; p<n_points; ++p)
		RightHandSide<dim>::vector_value (points[p],
				value_list[p]);
		}//RightHandSide<dim>::vector_value_list





template <int dim>
class ExactSolution : public Function<dim>
{
public:
	ExactSolution () : Function<dim>(dim+2) {} // changed to (darcy pressure, velocity, pressure)
	//originally (velocity,pressure,darcy pressure)

	virtual void vector_value (const Point<dim> &p,
			Vector<double>   &value) const;
};


template <int dim>
void
ExactSolution<dim>::vector_value (const Point<dim> &p,
		Vector<double>   &values) const
		{

	Assert (values.size() == dim+2,
			ExcDimensionMismatch (values.size(), dim+2));

	switch(dim)
	{
	case 2:
	{
		values(0) = p[1]*p[1] -2.0*p[1] + 2.0*p[1] + 2.0*p[0]; //0.0;
		values(1) = p[0]*p[0] - p[0] -2.0*p[1]; //-1.0;
		values(2) = -1.0*pow(p[0],2.0)*p[1] + p[0]*p[1] + pow(p[1],2.0)-4.0;
		values(3) = -1.0*pow(p[0],2.0)*p[1] + p[0]*p[1] + pow(p[1],2.0);
		return;
	}
	case 3:
	{
		values(0) = 0.0;
		values(1) = 0.0;
		values(2) = -1.0;
		values(3) = p[2]-1.0;
		values(4) = p[2]-1.0;
		break;
	}
	default:
		Assert(false,ExcNotImplemented());
	}

		}

template <class Matrix>
class InverseMatrix : public Subscriptor
{
public:
	InverseMatrix (const Matrix &m);
	void vmult (Vector<double> &dst, const Vector<double> &src) const;

private:
	const SmartPointer<const Matrix> matrix;
};

template <class Matrix>
InverseMatrix<Matrix>::InverseMatrix(const Matrix &m)
:
matrix (&m)
{}

template <class Matrix>
void InverseMatrix<Matrix>::vmult(Vector<double> &dst, const Vector<double> &src) const
{
	//matrix->vmult(dst,src);

	dst=0.0;

	SparseDirectUMFPACK direct;
	direct.initialize (*matrix);
	direct.vmult (dst, src);
};

template<class Preconditioner>
class SchurComplement : public Subscriptor
{
public:
	//SchurComplement (const BlockSparseMatrix<double> &A, const InverseMatrix<SparseMatrix<double>> &Minv);
	SchurComplement(const BlockSparseMatrix<double> &A,
			const Preconditioner &mg_prec,
			unsigned int const block_type);

	void vmult(Vector<double> &dst, const Vector<double> & src) const;
	void Tvmult(Vector<double>&dst, const Vector<double> &src) const;
private:
	const SmartPointer<const BlockSparseMatrix<double>> S;
//	const SmartPointer<const InverseMatrix<SparseMatrix<double >>> m_inverse;
	const Preconditioner mg_preconditioner;
	mutable Vector<double> tmp1, tmp2;
	unsigned int BP;
};

//template<class Preconditioner>
//SchurComplement<Preconditioner>::SchurComplement(const BlockSparseMatrix<double> &A,
//		const InverseMatrix<SparseMatrix<double>> &Minv) :
//	S (&A),
//	mg_preconditioner (&Minv),
//	tmp1 (A.block(0,0).m()),
//	tmp2 (A.block(1,1).m()),
//	BP(0)
//{}

template<class Preconditioner>
SchurComplement<Preconditioner>::SchurComplement(const BlockSparseMatrix<double> &A,
		const Preconditioner &mg_prec,
		unsigned int const block_type)
:
		S (&A),
		mg_preconditioner (mg_prec),
		tmp1(A.block(1,1).m()),
		tmp2(A.block(2,2).m()),
		BP (block_type)
{}

template<class Preconditioner>
void SchurComplement<Preconditioner>::vmult(Vector<double> &dst,const Vector<double> &src) const
{
	if (BP==0)
	{
	S->block(0,1).vmult(tmp1, src);
	mg_preconditioner.vmult(tmp2,tmp1);
	S->block(1,0).vmult(dst,tmp2);
	}
	else
	{
	S->block(1,2).vmult(tmp1, src);
	mg_preconditioner.vmult(tmp2,tmp1);
	S->block(2,1).vmult(dst,tmp2);
	}
}

template<class Preconditioner>
void SchurComplement<Preconditioner>::Tvmult(Vector<double> &dst,const Vector<double> &src) const
{
	vmult(dst,src);
}

template <class Matrix>
class BlockSchurPreconditioner : public Subscriptor
{
public:
	BlockSchurPreconditioner(const BlockSparseMatrix<double> &sys, const InverseMatrix<SparseMatrix<double>> &Minv,
			const IterativeInverse<Vector<double>> &schur);

	void vmult(BlockVector<double> &dst, BlockVector<double> &src) const;

private:
	const SmartPointer<const BlockSparseMatrix<double>> system;
	const SmartPointer<const InverseMatrix<SparseMatrix<double>>> a_preconditioner;
	const SmartPointer<const IterativeInverse<Vector<double>>> schur_operator;
	mutable Vector<double> tmp, tmp1, tmp2;
};

template <class Matrix>
BlockSchurPreconditioner<Matrix>::BlockSchurPreconditioner(const BlockSparseMatrix<double> &sys,
		const InverseMatrix<SparseMatrix<double>> &Minv,
		const IterativeInverse<Vector<double>> &schur)
:
system (&sys),
a_preconditioner (&Minv),
schur_operator (&schur),
tmp (sys.block(1,1).m()),
tmp1 (sys.block(0,0).m()),
tmp2 (sys.block(0,0).m())
{}

template <class Matrix>
void BlockSchurPreconditioner<Matrix>::vmult(BlockVector<double> &dst, BlockVector<double> &src) const
{
	a_preconditioner->vmult(dst.block(0), src.block(0));
	system->block(1,0).residual(tmp,dst.block(0),src.block(1));
	schur_operator->vmult(dst.block(1),tmp);
	system->block(0,1).vmult(tmp1,dst.block(1));
	a_preconditioner->vmult(tmp2,tmp1);
	dst.block(0)+=tmp2;
}

template<class Preconditioner_Darcy, class Preconditioner_Stokes, class Preconditioner_Schur>
class Block_MG_Preconditioner : public Subscriptor
{
public:
	Block_MG_Preconditioner(const BlockSparseMatrix<double> &S,
			const Preconditioner_Darcy &mg_darcy,
			const Preconditioner_Stokes &mg_stokes,
			const Preconditioner_Schur &schur_complement,
			const unsigned int pre,
			const unsigned int BP);

	void vmult(BlockVector<double> &dst, BlockVector<double> &src) const;

private:
	const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
	const Preconditioner_Darcy darcy_pre;
	const Preconditioner_Stokes stokes_pre;
	const Preconditioner_Schur schur_preconditioner;
	mutable Vector<double> tmp;
	mutable Vector<double> tmp1;
	mutable Vector<double> tmp2;
	const unsigned int p_type;
	const unsigned int block_type;

};

template<class Preconditioner_Darcy, class Preconditioner_Stokes, class Preconditioner_Schur>
Block_MG_Preconditioner<Preconditioner_Darcy,Preconditioner_Stokes,Preconditioner_Schur>::
Block_MG_Preconditioner(const BlockSparseMatrix<double> &S,
		const Preconditioner_Darcy &mg_darcy,
		const Preconditioner_Stokes &mg_stokes,
		const Preconditioner_Schur &schur_complement,
		const unsigned int pre,
		const unsigned int BP)
		:
		system_matrix (&S),
		darcy_pre (mg_darcy),
		stokes_pre (mg_stokes),
		schur_preconditioner (schur_complement),
		tmp (S.block(2,2).m()),
		tmp1 (S.block(1,1).m()),
		tmp2 (S.block(1,1).m()),
		p_type(pre),
		block_type(BP)
		{}

template<class Preconditioner_Darcy, class Preconditioner_Stokes, class Preconditioner_Schur>
void Block_MG_Preconditioner<Preconditioner_Darcy,Preconditioner_Stokes,Preconditioner_Schur>::
vmult(BlockVector<double> &dst, BlockVector<double> &src) const
{
	dst=0;
//	switch(block_type)
//	{
//	case 2:
//	{
//	switch (p_type)
//	{
//	// Block diagonal
////	case 1:
////	{
////		darcy_pre->vmult(dst.block(0),src.block(0));
////		schur_preconditioner->vmult (dst.block(1),src.block(1));
////		break;
////	}
////	// Block lower triangular
////	case 2:
////	{
////		a_preconditioner->vmult(dst.block(0), src.block(0));
////		system_matrix->block(1,0).residual(tmp, dst.block(0), src.block(1));
////		schur_preconditioner->vmult(dst.block(1), tmp);
////		break;
////	}
////	// Constraint
////	case 3:
////	{
////		a_preconditioner->vmult(dst.block(0), src.block(0));
////		system_matrix->block(1,0).residual(tmp,dst.block(0),src.block(1));
////		schur_preconditioner->vmult(dst.block(1),tmp);
////		system_matrix->block(0,1).vmult(tmp1,dst.block(1));
////		a_preconditioner->vmult(tmp2,tmp1);
////		dst.block(0)+=tmp2;
////	}
//	}
//	break;
//	}
//	case 3:
//	{
	switch (p_type)
	{
	// Block diagonal
	case 1:
	{
		darcy_pre.vmult(dst.block(0),src.block(0));
		stokes_pre.vmult(dst.block(1),src.block(1));
		schur_preconditioner.vmult (dst.block(2),src.block(2));

		break;
	}
	// Block lower triangular
	case 2:
	{
	darcy_pre.vmult(dst.block(0), src.block(0));
	stokes_pre.vmult(dst.block(1), src.block(1));
	system_matrix->block(2,1).residual(tmp, dst.block(1), src.block(2));
	schur_preconditioner.vmult(dst.block(2), tmp);
	break;
	}
	// Constraint
	case 3:
	{
	darcy_pre.vmult(dst.block(0), src.block(0));
	stokes_pre.vmult(dst.block(1),src.block(1));
	system_matrix->block(2,1).residual(tmp,dst.block(1),src.block(2));
	schur_preconditioner.vmult(dst.block(2),tmp);
	system_matrix->block(1,2).vmult(tmp1,dst.block(2));
	stokes_pre.vmult(tmp2,tmp1);
	dst.block(1)-=tmp2;
	}
	}
//	}
//	}
}

template <class Matrix>
class BlockPreconditioner : public Subscriptor
{
public:
	BlockPreconditioner(const BlockSparseMatrix<double> &S,
			const InverseMatrix<SparseMatrix<double>> &P,
			const InverseMatrix<SparseMatrix<double>> &Q,
			const unsigned int pre,
			const unsigned int BP);


	BlockPreconditioner(const BlockSparseMatrix<double> &S,
			const InverseMatrix<SparseMatrix<double>> &P1,
			const InverseMatrix<SparseMatrix<double>> &P2,
			const InverseMatrix<SparseMatrix<double>> &Ps,
			const unsigned int pre,
			const unsigned int BP);


	void vmult(BlockVector<double>       &dst,
			const BlockVector<double> &src) const;

private:
	const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
	const SmartPointer<const InverseMatrix<SparseMatrix<double>>> a_preconditioner;
	const SmartPointer<const InverseMatrix<SparseMatrix<double>>> b_preconditioner;
	const SmartPointer<const InverseMatrix<SparseMatrix<double>>> schur_preconditioner;
	mutable Vector<double> tmp;
	mutable Vector<double> tmp1;
	mutable Vector<double> tmp2;
	const unsigned int p_type;
	const unsigned int block_type;
};

template <class Matrix>
BlockPreconditioner<Matrix>::BlockPreconditioner(const BlockSparseMatrix<double> &S,
		const InverseMatrix <SparseMatrix<double>> &P_inv,
		const InverseMatrix <SparseMatrix<double>> &Q_inv,
		const unsigned int pre,
		const unsigned int BP)
		:
		system_matrix (&S),
		a_preconditioner (&P_inv),
		schur_preconditioner (&Q_inv),
		tmp (S.block(1,1).m()),
		tmp1 (S.block(0,0).m()),
		tmp2 (S.block(0,0).m()),
		p_type(pre),
		block_type(BP)
		{}


template <class Matrix>
BlockPreconditioner<Matrix>::BlockPreconditioner(const BlockSparseMatrix<double> &S,
		const InverseMatrix <SparseMatrix<double>> &P_inv,
		const InverseMatrix <SparseMatrix<double>> &Q_inv,
		const InverseMatrix <SparseMatrix<double>> &R_inv,
		const unsigned int pre,
		const unsigned int BP)
		:
		system_matrix (&S),
		a_preconditioner (&P_inv),
		b_preconditioner (&Q_inv),
		schur_preconditioner (&R_inv),
		tmp (S.block(2,2).m()),
		tmp1 (S.block(1,1).m()),
		tmp2 (S.block(1,1).m()),
		p_type(pre),
		block_type(BP)
		{}


template <class Matrix>
void BlockPreconditioner<Matrix>::vmult(BlockVector<double>       &dst,
		const BlockVector<double> &src) const
{
	switch(block_type)
	{
	case 2:
	{
	switch (p_type)
	{
	// Block diagonal
	case 1:
	{
		a_preconditioner->vmult(dst.block(0),src.block(0));
		schur_preconditioner->vmult (dst.block(1),src.block(1));
		break;
	}
	// Block lower triangular
	case 2:
	{
		a_preconditioner->vmult(dst.block(0), src.block(0));
		system_matrix->block(1,0).residual(tmp, dst.block(0), src.block(1));
		schur_preconditioner->vmult(dst.block(1), tmp);
		break;
	}
	// Constraint
	case 3:
	{
		a_preconditioner->vmult(dst.block(0), src.block(0));
		system_matrix->block(1,0).residual(tmp,dst.block(0),src.block(1));
		schur_preconditioner->vmult(dst.block(1),tmp);
		system_matrix->block(0,1).vmult(tmp1,dst.block(1));
		a_preconditioner->vmult(tmp2,tmp1);
		dst.block(0)+=tmp2;
	}
	}
	break;
	}
	case 3:
	{
	switch (p_type)
	{
	// Block diagonal
	case 1:
	{
		a_preconditioner->vmult(dst.block(0),src.block(0));
		b_preconditioner->vmult(dst.block(1),src.block(1));
		schur_preconditioner->vmult (dst.block(2),src.block(2));
		break;
	}
	// Block lower triangular
	case 2:
	{
	a_preconditioner->vmult(dst.block(0), src.block(0));
	b_preconditioner->vmult(dst.block(1), src.block(1));
	system_matrix->block(2,1).residual(tmp, dst.block(1), src.block(2));
	schur_preconditioner->vmult(dst.block(2), tmp);
	break;
	}
	// Constraint
	case 3:
	{
	a_preconditioner->vmult(dst.block(0), src.block(0));
	b_preconditioner->vmult(dst.block(1),src.block(1));
	system_matrix->block(2,1).residual(tmp,dst.block(1),src.block(2));
	schur_preconditioner->vmult(dst.block(2),tmp);
	system_matrix->block(1,2).vmult(tmp1,dst.block(2));
	b_preconditioner->vmult(tmp2,tmp1);
	dst.block(1)-=tmp2;
	}
	}
	}
	}
};



// @sect3{The <code>StokesDarcy</code> implementation}

// @sect4{Constructors and helper functions}

// Let's now get to the implementation of the primary class of this
// program. The first few functions are the constructor and the helper
// functions that can be used to determine which part of the domain a cell
// is in. Given the discussion of these topics in the introduction, their
// implementation is rather obvious. In the constructor, note that we have
// to construct the hp::FECollection object from the base elements for
// Stokes and darcy; using the hp::FECollection::push_back function
// assigns them spots zero and one in this collection, an order that we have
// to remember and use consistently in the rest of the program.
template <int dim>
StokesDarcy<dim>::
StokesDarcy(const unsigned int stokes_degree, const unsigned int darcy_degree,
		const unsigned int pre_type)
		:
		stokes_degree (stokes_degree),
		darcy_degree (darcy_degree),
		triangulation (Triangulation<dim>::maximum_smoothing),
		stokes_fe (FE_Q<dim>(stokes_degree+1), dim, //Velocity vector
				FE_Q<dim>(stokes_degree), 1,     //Pressure scalar
				FE_Nothing<dim>(), 1),           //Darcy pressure zero extension
		darcy_fe (FE_Nothing<dim>(), dim,    //Velocity extension
				FE_Nothing<dim>(), 1,        //Stokes pressure extension
				FE_Q<dim>(darcy_degree), 1), //Scalar
		stokes_mg_fe (FE_Q<dim>(stokes_degree+1), dim, //Velocity vector
				FE_Nothing<dim>(), 1,     //Pressure scalar zero extension since we only need (\grad u,\grad v)
				FE_Nothing<dim>(), 1),
		dof_handler (triangulation),
		viscosity(1.0),
		mu (1.0),
		G (1.0),
		None(-1.0),
		pre_type(pre_type),
		darcy_tri (Triangulation<dim>::maximum_smoothing),
		darcy_dof_handler(darcy_tri),
		stokes_tri(Triangulation<dim>::maximum_smoothing),
		stokes_dof_handler(stokes_tri)
	{
	fe_collection.push_back (stokes_fe);
	fe_collection.push_back (darcy_fe);
	}




template <int dim>
bool
StokesDarcy<dim>::
cell_is_in_fluid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
{
	return (cell->material_id() == fluid_domain_id);
}

template <int dim>
bool
StokesDarcy<dim>::
cell_is_in_fluid_domain_mg (const typename DoFHandler<dim>::cell_iterator &cell)
{
	return (cell->material_id() == fluid_domain_id);
}


template <int dim>
bool
StokesDarcy<dim>::
cell_is_in_darcy_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
{
	return (cell->material_id() == darcy_domain_id);
}

template <int dim>
bool
StokesDarcy<dim>::
cell_is_in_darcy_domain_mg (const typename DoFHandler<dim>::cell_iterator &cell)
{
	return (cell->material_id() == darcy_domain_id);
}


// @sect4{Meshes and assigning subdomains}

// The next pair of functions deals with generating a mesh and making sure
// all flags that denote subdomains are correct. <code>make_grid</code>, as
// discussed in the introduction, generates an $8\times 8$ mesh (or an
// $8\times 8\times 8$ mesh in 3d) to make sure that each coarse mesh cell
// is completely within one of the subdomains. After generating this mesh,
// we loop over its boundary and set the boundary indicator to one at the
// top boundary, the only place where we set nonzero Dirichlet boundary
// conditions. After this, we loop again over all cells to set the material
// indicator &mdash; used to denote which part of the domain we are in, to
// either the fluid or darcy indicator.
template <int dim>
void
StokesDarcy<dim>::make_grid ()
{
	GridGenerator::subdivided_hyper_cube (triangulation, 4, 0, 2);

	//Multigrid sub grids, need to make the end points between the two dependent for easier changing
	switch (dim)
		{

		case 2 :
		{
		std::vector<unsigned int> repetitions (2);
		repetitions[0]=4;
		repetitions[1]=2;
		Point<dim> p1 (0.0,1.0);
		Point<dim> p2 (2.0,2.0);
		Point<dim> p3 (0.0,0.0);
		Point<dim> p4 (2.0,1.0);

		GridGenerator::subdivided_hyper_rectangle (stokes_tri,repetitions , p1, p2);
		GridGenerator::subdivided_hyper_rectangle (darcy_tri, repetitions, p3, p4);
		break;
		}
		case 3 :
		{
		std::vector<unsigned int> repetitions (3);
		repetitions[0]=4;
		repetitions[1]=4;
		repetitions[2]=2;
		Point<dim> p1 (0.0,0.0,1.0);
		Point<dim> p2 (2.0,2.0,2.0);
		Point<dim> p3 (0.0,0.0,0.0);
		Point<dim> p4 (2.0,2.0,1.0);
		GridGenerator::subdivided_hyper_rectangle (stokes_tri,repetitions , p1, p2);
		GridGenerator::subdivided_hyper_rectangle (darcy_tri, repetitions, p3, p4);
		break;
		}

		}


	for (typename Triangulation<dim>::active_cell_iterator
			cell = dof_handler.begin_active();
			cell != dof_handler.end(); ++cell)
		if((cell->center()[dim-1] > 1.0))
		{
			cell->set_material_id (fluid_domain_id);
		}
		else
		{
			cell->set_material_id (darcy_domain_id);
		}

	for (typename Triangulation<dim>::active_cell_iterator
			cell = triangulation.begin_active();
			cell != triangulation.end(); ++cell)
		for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
			if ((cell->face(f)->at_boundary()) && (cell->material_id() == fluid_domain_id))
			{
				cell->face(f)->set_all_boundary_indicators(1); //Dirichlet b.c everywhere
			}

	//These boundary conditions right now are problem dependent and only for 2D !!!
	// MG Stokes bndry conditions
	for (typename Triangulation<dim>::active_cell_iterator
			cell = stokes_tri.begin_active();
			cell != stokes_tri.end(); ++cell)
		for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
			if ((cell->face(f)->at_boundary()) && (cell->face(f)->center()[dim-1] != 1 )
					&& ( cell->face(f)->center()[0] == 0 || cell->face(f)->center()[0] ==2 ) )
			{
				cell->face(f)->set_all_boundary_indicators(1); //Dirichlet b.c everywhere
			}

	//MG darcy bndry condition set, loop over all cells of darcy triangulation
	for (typename Triangulation<dim>::active_cell_iterator
			cell = darcy_tri.begin_active();
			cell != darcy_tri.end(); ++cell)
		for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
			if ((cell->face(f)->at_boundary()) && (cell->face(f)->center()[dim-1] != 1)
					&& ( cell->face(f)->center()[0] == 0 || cell->face(f)->center()[0] ==2 ))
			{
				cell->face(f)->set_all_boundary_indicators(1); //Dirichlet b.c everywhere
			}

	std::ofstream out ("full-grid.eps");
	GridOut grid_out;
	grid_out.write_eps (triangulation, out);

	triangulation.refine_global(1);
	stokes_tri.refine_global(1);
	darcy_tri.refine_global(1);

	/*
		std::ofstream out ("darcy-grid.eps");
		GridOut grid_out;
		grid_out.write_eps (darcy_tri, out);

		std::ofstream out1 ("stokes-grid.eps");
		GridOut grid_out1;
		grid_out1.write_eps (stokes_tri, out1);
	*/
}


// The second part of this pair of functions determines which finite element
// to use on each cell. Above we have set the material indicator for each
// coarse mesh cell, and as mentioned in the introduction, this information
// is inherited from mother to child cell upon mesh refinement.
//
// In other words, whenever we have refined (or created) the mesh, we can
// rely on the material indicators to be a correct description of which part
// of the domain a cell is in. We then use this to set the active FE index
// of the cell to the corresponding element of the hp::FECollection member
// variable of this class: zero for fluid cells, one for darcy cells.
template <int dim>
void
StokesDarcy<dim>::set_active_fe_indices ()
{
	for (typename hp::DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active();
			cell != dof_handler.end(); ++cell)
	{
		if (cell_is_in_fluid_domain(cell))
			cell->set_active_fe_index (0);
		else if (cell_is_in_darcy_domain(cell))
			cell->set_active_fe_index (1);
		else
			Assert (false, ExcNotImplemented());
	}
}


// @sect4{<code>StokesDarcy::setup_dofs</code>}

// The next step is to setup the data structures for the linear system. To
// this end, we first have to set the active FE indices with the function
// immediately above, then distribute degrees of freedom, and then determine
// constraints on the linear system. The latter includes hanging node
// constraints as usual, but also the inhomogeneous boundary values at the
// top fluid boundary, and zero boundary values along the perimeter of the
// darcy subdomain.


template <int dim>
void
StokesDarcy<dim>::setup_dofs (const unsigned int block_pattern)
{
	dof_handler.distribute_dofs(fe_collection);

	set_active_fe_indices ();
	dof_handler.distribute_dofs (fe_collection);

	std::cout << "Number of Dofs: " << dof_handler.n_dofs() << "\n";

	std::vector<unsigned int> block_component(dim+2,1);
	block_component[dim]=2;
	block_component[dim+1]=0;

	DoFRenumbering::component_wise(dof_handler, block_component);

	{
		constraints.clear ();
		DoFTools::make_hanging_node_constraints (dof_handler, constraints);

		const FEValuesExtractors::Vector velocities(0);

		VectorTools::interpolate_boundary_values (dof_handler,
				1,
				StokesBoundaryValues<dim>(),
				constraints,
				fe_collection.component_mask(velocities));

		//  std::cout << "velocities" << fe_collection.component_mask(velocities) << "\n";


		const FEValuesExtractors::Scalar darcy_pressure(dim+1);                    //scalar pressure
		VectorTools::interpolate_boundary_values (dof_handler,
				0,
				DarcyBoundaryValues<dim>(),
				constraints,
				fe_collection.component_mask(darcy_pressure));

		//std::cout << "darcy pressure" << fe_collection.component_mask(darcy_pressure) << "\n";

	}


	// At the end of all this, we can declare to the constraints object that
	// we now have all constraints ready to go and that the object can rebuild
	// its internal data structures for better efficiency:
	constraints.close ();

	std::vector<types::global_dof_index> dofs_per_block (3);
	DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
	const unsigned int n_uf = dofs_per_block[1],
			n_pf = dofs_per_block[2],
			n_pd = dofs_per_block[0];

	std::cout << "   Number of active cells: "
			<< triangulation.n_active_cells()
			<< "\n"
			<< "   Number of degrees of freedom: "
			<< dof_handler.n_dofs()
			<< " (" << n_uf << '+' << n_pf << '+' << n_pd << ')'
			<< "\n";

	// In the rest of this function we create a sparsity pattern as discussed
	// extensively in the introduction, and use it to initialize the matrix;
	// then also set vectors to their correct sizes:

	switch (block_pattern)
	{
	case 2 :
	{
	{
		BlockCompressedSimpleSparsityPattern csp (2,2);
		csp.block(0,0).reinit (n_pd+n_uf,n_pd+n_uf);
		csp.block(1,1).reinit (n_pf,n_pf);
		csp.block(0,1).reinit (n_pd+n_uf,n_pf);
		csp.block(1,0).reinit (n_pf,n_uf+n_pd);

		csp.collect_sizes();

		Table<2,DoFTools::Coupling> cell_coupling (fe_collection.n_components(),
				fe_collection.n_components());
		Table<2,DoFTools::Coupling> face_coupling (fe_collection.n_components(),
				fe_collection.n_components());

		for (unsigned int c=0; c<fe_collection.n_components(); ++c)
			for (unsigned int d=0; d<fe_collection.n_components(); ++d)
			{
				if (((c<dim+1) && (d<dim+1)
						&& !((c==dim) && (d==dim)))
						||
						((c>=dim+1) && (d>=dim+1)))
					cell_coupling[c][d] = DoFTools::always;

				if (((c<dim+1) && (d<dim+1)) ||
						((c< dim+1)&& (d>=dim+1)) ||
						((c >=dim+1)&& (d < dim +1)))
					face_coupling[c][d] = DoFTools::always;
			}

		DoFTools::make_flux_sparsity_pattern (dof_handler, csp,
				cell_coupling, face_coupling);
		constraints.condense (csp);
		sparsity_pattern.copy_from (csp);
	}

	system_matrix.reinit (sparsity_pattern);

	solution.reinit (2);
	solution.block(0).reinit (n_pd+n_uf);
	solution.block(1).reinit (n_pf);
	solution.collect_sizes();

	system_rhs.reinit (2);
	system_rhs.block(0).reinit (n_pd+n_uf);
	system_rhs.block(1).reinit (n_pf);
	system_rhs.collect_sizes();
	break;
	}

	case 3 :
	{
	{
		BlockCompressedSimpleSparsityPattern csp (3,3);
		csp.block(0,0).reinit (n_pd,n_pd);
		csp.block(1,1).reinit (n_uf,n_uf);
		csp.block(2,2).reinit (n_pf,n_pf);
		csp.block(0,1).reinit (n_pd,n_uf);
		csp.block(1,0).reinit (n_uf,n_pd);
		csp.block(0,2).reinit (n_pd,n_pf);
		csp.block(2,0).reinit (n_pf,n_pd);
		csp.block(2,1).reinit (n_pf,n_uf);
		csp.block(1,2).reinit (n_uf,n_pf);

		csp.collect_sizes();

		Table<2,DoFTools::Coupling> cell_coupling (fe_collection.n_components(),
			fe_collection.n_components());
		Table<2,DoFTools::Coupling> face_coupling (fe_collection.n_components(),
			fe_collection.n_components());

		for (unsigned int c=0; c<fe_collection.n_components(); ++c)
			for (unsigned int d=0; d<fe_collection.n_components(); ++d)
				{
					if (((c<dim+1) && (d<dim+1)
							&& !((c==dim) && (d==dim)))
							||
							((c>=dim+1) && (d>=dim+1)))
						cell_coupling[c][d] = DoFTools::always;

					if (((c<dim+1) && (d<dim+1)) ||
							((c< dim+1)&& (d>=dim+1)) ||
							((c >=dim+1)&& (d < dim +1)))
						face_coupling[c][d] = DoFTools::always;
				}

		DoFTools::make_flux_sparsity_pattern (dof_handler, csp,	cell_coupling, face_coupling);
		constraints.condense (csp);
		sparsity_pattern.copy_from (csp);
	}

	system_matrix.reinit (sparsity_pattern);

	solution.reinit (3);
	solution.block(0).reinit (n_pd);
	solution.block(1).reinit (n_uf);
	solution.block(2).reinit (n_pf);
	solution.collect_sizes();

	system_rhs.reinit (3);
	system_rhs.block(0).reinit (n_pd);
	system_rhs.block(1).reinit (n_uf);
	system_rhs.block(2).reinit (n_pf);
	system_rhs.collect_sizes();
	break;
	}

	default :
	{
		Assert (false, ExcNotImplemented());
	}
	}

}



// @sect4{<code>StokesDarcy::assemble_system</code>}

// Following is the central function of this program: the one that assembles
// the linear system. It has a long section of setting up auxiliary
// functions at the beginning: from creating the quadrature formulas and
// setting up the FEValues, FEFaceValues and FESubfaceValues objects
// necessary to integrate the cell terms as well as the interface terms for
// the case where cells along the interface come together at same size or
// with differing levels of refinement...
template <int dim>
void StokesDarcy<dim>::assemble_system ()
{
	system_matrix=0;
	system_rhs=0;

	const QGauss<dim> stokes_quadrature(stokes_degree+2);
	const QGauss<dim> darcy_quadrature(darcy_degree+1);

	hp::QCollection<dim>  q_collection;
	q_collection.push_back (stokes_quadrature);
	q_collection.push_back (darcy_quadrature);

	hp::FEValues<dim> hp_fe_values (fe_collection, q_collection,
			update_values    |
			update_quadrature_points  |
			update_JxW_values |
			update_gradients);

	const QGauss<dim-1> common_face_quadrature(std::max (stokes_degree+2,
			darcy_degree+2));

	FEFaceValues<dim>    stokes_fe_face_values (stokes_fe,
			common_face_quadrature,
			update_JxW_values |
			update_normal_vectors |
			update_values |
			update_gradients);
	FEFaceValues<dim>    darcy_fe_face_values (darcy_fe,
			common_face_quadrature,
			update_values);
	FESubfaceValues<dim> stokes_fe_subface_values (stokes_fe,
			common_face_quadrature,
			update_JxW_values |
			update_normal_vectors |
			update_values |
			update_gradients);
	FESubfaceValues<dim> darcy_fe_subface_values (darcy_fe,
			common_face_quadrature,
			update_values);

	// ...to objects that are needed to describe the local contributions to
	// the global linear system...
	const unsigned int        stokes_dofs_per_cell     = stokes_fe.dofs_per_cell;
	const unsigned int        darcy_dofs_per_cell = darcy_fe.dofs_per_cell;

	const unsigned int        n_q_points_stokes = stokes_quadrature.size();

	FullMatrix<double>        local_matrix;
	FullMatrix<double>        local_interface_matrix (darcy_dofs_per_cell,
			stokes_dofs_per_cell);
	FullMatrix<double>        local_interface_matrix_velocity_pressure(stokes_dofs_per_cell,
			darcy_dofs_per_cell);
	FullMatrix<double>        local_interface_matrix_pressure_velocity(darcy_dofs_per_cell,
			stokes_dofs_per_cell);
	FullMatrix<double>        local_interface_matrix_velocity_velocity(stokes_dofs_per_cell,
			stokes_dofs_per_cell);


	Vector<double>            local_rhs;

	std::vector<types::global_dof_index> local_dof_indices;
	std::vector<types::global_dof_index> neighbor_dof_indices (stokes_dofs_per_cell);

	const RightHandSide<dim>  right_hand_side;
	std::vector<Vector<double>> rhs_values(n_q_points_stokes,
			Vector<double>(dim+1));

	const DarcyRightHandSide<dim> darcy_right_hand_side;

	// ...to variables that allow us to extract certain components of the
	// shape functions and cache their values rather than having to recompute
	// them at every quadrature point:
	const FEValuesExtractors::Vector     velocities (0);
	const FEValuesExtractors::Scalar     pressure (dim);
	const FEValuesExtractors::Scalar     darcy_pressure (dim+1);

	std::vector<SymmetricTensor<2,dim> > stokes_symgrad_phi_u (stokes_dofs_per_cell); //Stokes symmetric gradient values
	std::vector<double>                  stokes_div_phi_u     (stokes_dofs_per_cell); //Stokes divergence of u values
	std::vector<Tensor<1,dim>>           stokes_phi_u         (stokes_dofs_per_cell);  //Stokes u values
	std::vector<double>                  stokes_phi_p         (stokes_dofs_per_cell); //Stokes pressure values

	std::vector<double>                  darcy_phi_p          (darcy_dofs_per_cell);  //Darcy pressure values
	std::vector<Tensor<1,dim>>           darcy_grad_phi       (darcy_dofs_per_cell);  //Darcy gradient p values

	// Then comes the main loop over all cells and, as in step-27, the
	// initialization of the hp::FEValues object for the current cell and the
	// extraction of a FEValues object that is appropriate for the current
	// cell:
	typename hp::DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{

		hp_fe_values.reinit (cell);


		const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

		local_matrix.reinit (cell->get_fe().dofs_per_cell,
				cell->get_fe().dofs_per_cell);
		local_rhs.reinit (cell->get_fe().dofs_per_cell);

		// With all of this done, we continue to assemble the cell terms for
		// cells that are part of the Stokes and elastic regions. While we
		// could in principle do this in one formula, in effect implementing
		// the one bilinear form stated in the introduction, we realize that
		// our finite element spaces are chosen in such a way that on each
		// cell, one set of variables (either velocities and pressure, or
		// darcy_pressure) are always zero, and consequently a more efficient
		// way of computing local integrals is to do only what's necessary
		// based on an <code>if</code> clause that tests which part of the
		// domain we are in.
		//
		// The actual computation of the local matrix is the same as in
		// step-22 as well as that given in the @ref vector_valued
		// documentation module for the darcy equations:



		//std::cout << "cell " << cell << "cell center : " << cell->center()[0] << cell->center()[1] << "\n";

		if (cell_is_in_fluid_domain (cell))
		{

			const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
			Assert (dofs_per_cell == stokes_dofs_per_cell,
					ExcInternalError());


			local_matrix=0;
			local_rhs=0;

			right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
					rhs_values);


			//Stokes sub-domain
			for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
			{
				for (unsigned int k=0; k<dofs_per_cell; ++k)
				{
					stokes_symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
					stokes_div_phi_u[k]     = fe_values[velocities].divergence (k, q);
					stokes_phi_u[k]         = fe_values[velocities].value(k,q);
					stokes_phi_p[k]         = fe_values[pressure].value (k, q);


				}

				for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
					for (unsigned int j=0; j<dofs_per_cell; ++j)
					{
						local_matrix(i,j) += (2 * viscosity * stokes_symgrad_phi_u[i] * stokes_symgrad_phi_u[j]
						                                                                                     - stokes_div_phi_u[i] * stokes_phi_p[j]
						                                                                                                                          - stokes_phi_p[i] * stokes_div_phi_u[j])
						                                                                                                                          * fe_values.JxW(q);
					}
					//add the local rhs for the velocity components and zero out for the pressure

					const unsigned int component_i =
							stokes_fe.system_to_component_index(i).first;
					local_rhs(i) += fe_values.shape_value(i,q) *
							rhs_values[q](component_i) *
							fe_values.JxW(q);


				}
			}
		}
		else
		{
			const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
			Assert (dofs_per_cell == darcy_dofs_per_cell,
					ExcInternalError());

			for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
			{
				for (unsigned int k=0; k<dofs_per_cell; ++k)
				{
					darcy_grad_phi[k] = fe_values[darcy_pressure].gradient (k, q);
					darcy_phi_p[k] = fe_values[darcy_pressure].value(k,q);

				}
				//Darcy sub-domain
				for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
					for (unsigned int j=0; j<dofs_per_cell; ++j)
					{
						local_matrix(i,j) += (darcy_grad_phi[i]*darcy_grad_phi[j])*fe_values.JxW(q);
					}

					//add the local rhs for the darcy pressure components


					local_rhs(i) += (fe_values.shape_value(i,q) * darcy_right_hand_side.value(fe_values.quadrature_point(q))*fe_values.JxW(q));
					//std::cout << "local_rhs " << local_rhs(i) << "i" << i << "\n";



				}
			}
		}
		//Darcy region


		// Once we have the contributions from cell integrals, we copy them
		// into the global matrix (taking care of constraints right away,
		// through the ConstraintMatrix::distribute_local_to_global
		// function). Note that we have not written anything into the
		// <code>local_rhs</code> variable, though we still need to pass it
		// along since the elimination of nonzero boundary values requires the
		// modification of local and consequently also global right hand side
		// values:

		local_dof_indices.resize (cell->get_fe().dofs_per_cell);
		cell->get_dof_indices (local_dof_indices);
		//for(unsigned int i=0; i< cell->get_fe().dofs_per_cell; ++i)
		//    std::cout << "toto " << local_dof_indices[i] << "        i     " << i <<"\n";
		constraints.distribute_local_to_global (local_matrix, local_rhs,
				local_dof_indices,
				system_matrix, system_rhs);

		// The more interesting part of this function is where we see about
		// face terms along the interface between the two subdomains. To this
		// end, we first have to make sure that we only assemble them once
		// even though a loop over all faces of all cells would encounter each
		// part of the interface twice. We arbitrarily make the decision that
		// we will only evaluate interface terms if the current cell is part
		// of the darcy subdomain and if, consequently, a face is not at the
		// boundary and the potential neighbor behind it is part of the fluid
		// domain. Let's start with these conditions:
		if (cell_is_in_darcy_domain (cell))
			for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
				if (cell->at_boundary(f) == false)
				{
					// At this point we know that the current cell is a candidate
					// for integration and that a neighbor behind face
					// <code>f</code> exists. There are now three possibilities:
					//
					// - The neighbor is at the same refinement level and has no
					//   children.
					// - The neighbor has children.
					// - The neighbor is coarser.
					//
					// In all three cases, we are only interested in it if it is
					// part of the fluid subdomain. So let us start with the first
					// and simplest case: if the neighbor is at the same level,
					// has no children, and is a fluid cell, then the two cells
					// share a boundary that is part of the interface along which
					// we want to integrate interface terms. All we have to do is
					// initialize two FEFaceValues object with the current face
					// and the face of the neighboring cell (note how we find out
					// which face of the neighboring cell borders on the current
					// cell) and pass things off to the function that evaluates
					// the interface terms (the third through fifth arguments to
					// this function provide it with scratch arrays). The result
					// is then again copied into the global matrix, using a
					// function that knows that the DoF indices of rows and
					// columns of the local matrix result from different cells:
					if ((cell->neighbor(f)->level() == cell->level())
							&&
							(cell->neighbor(f)->has_children() == false)
							&&
							cell_is_in_fluid_domain (cell->neighbor(f)))
					{
						darcy_fe_face_values.reinit (cell, f);
						stokes_fe_face_values.reinit (cell->neighbor(f),
								cell->neighbor_of_neighbor(f));


						assemble_interface_term (darcy_fe_face_values,
								stokes_fe_face_values,
								darcy_phi_p,
								stokes_symgrad_phi_u,
								stokes_phi_u,
								stokes_phi_p,
								local_interface_matrix_velocity_pressure,
								local_interface_matrix_pressure_velocity,
								local_interface_matrix_velocity_velocity);

						cell->neighbor(f)->get_dof_indices (neighbor_dof_indices);

						constraints.distribute_local_to_global(local_interface_matrix_pressure_velocity,
								local_dof_indices,
								neighbor_dof_indices,
								system_matrix);

						constraints.distribute_local_to_global(local_interface_matrix_velocity_pressure,
								neighbor_dof_indices,
								local_dof_indices,
								system_matrix);

						constraints.distribute_local_to_global(local_interface_matrix_velocity_velocity,
								neighbor_dof_indices,
								neighbor_dof_indices,
								system_matrix);


					}

					// The second case is if the neighbor has further children. In
					// that case, we have to loop over all the children of the
					// neighbor to see if they are part of the fluid subdomain. If
					// they are, then we integrate over the common interface,
					// which is a face for the neighbor and a subface of the
					// current cell, requiring us to use an FEFaceValues for the
					// neighbor and an FESubfaceValues for the current cell:
					else if ((cell->neighbor(f)->level() == cell->level())
							&&
							(cell->neighbor(f)->has_children() == true))
					{
						for (unsigned int subface=0;
								subface<cell->face(f)->n_children();
								++subface)
							if (cell_is_in_fluid_domain (cell->neighbor_child_on_subface
									(f, subface)))
							{
								darcy_fe_subface_values.reinit (cell,
										f,
										subface);
								stokes_fe_face_values.reinit (cell->neighbor_child_on_subface (f, subface),
										cell->neighbor_of_neighbor(f));

								assemble_interface_term (darcy_fe_subface_values,
										stokes_fe_face_values,
										darcy_phi_p,
										stokes_symgrad_phi_u,
										stokes_phi_u,
										stokes_phi_p,
										local_interface_matrix_velocity_pressure,
										local_interface_matrix_pressure_velocity,
										local_interface_matrix_velocity_velocity);

								cell->neighbor_child_on_subface (f, subface)
                        						  ->get_dof_indices (neighbor_dof_indices);

								constraints.distribute_local_to_global(local_interface_matrix_pressure_velocity,
										local_dof_indices,
										neighbor_dof_indices,
										system_matrix);
								constraints.distribute_local_to_global(local_interface_matrix_velocity_pressure,
										neighbor_dof_indices,
										local_dof_indices,
										system_matrix);
								constraints.distribute_local_to_global(local_interface_matrix_velocity_velocity,
										neighbor_dof_indices,
										neighbor_dof_indices,
										system_matrix);
							}
					}

					// The last option is that the neighbor is coarser. In that
					// case we have to use an FESubfaceValues object for the
					// neighbor and a FEFaceValues for the current cell; the rest
					// is the same as before:
					else if (cell->neighbor_is_coarser(f)
							&&
							cell_is_in_fluid_domain(cell->neighbor(f)))
					{
						darcy_fe_face_values.reinit (cell, f);
						stokes_fe_subface_values.reinit (cell->neighbor(f),
								cell->neighbor_of_coarser_neighbor(f).first,
								cell->neighbor_of_coarser_neighbor(f).second);

						assemble_interface_term (darcy_fe_face_values,
								stokes_fe_subface_values,
								darcy_phi_p,
								stokes_symgrad_phi_u,
								stokes_phi_u,
								stokes_phi_p,
								local_interface_matrix_velocity_pressure,
								local_interface_matrix_pressure_velocity,
								local_interface_matrix_velocity_velocity);

						cell->neighbor(f)->get_dof_indices (neighbor_dof_indices);
						constraints.distribute_local_to_global(local_interface_matrix_pressure_velocity,
								local_dof_indices,
								neighbor_dof_indices,
								system_matrix);

						constraints.distribute_local_to_global(local_interface_matrix_velocity_pressure,
								neighbor_dof_indices,
								local_dof_indices,
								system_matrix);

						constraints.distribute_local_to_global(local_interface_matrix_velocity_velocity,
								neighbor_dof_indices,
								neighbor_dof_indices,
								system_matrix);



					}
				}
	}
}



// In the function that assembles the global system, we passed computing
// interface terms to a separate function we discuss here. The key is that
// even though we can't predict the combination of FEFaceValues and
// FESubfaceValues objects, they are both derived from the FEFaceValuesBase
// class and consequently we don't have to care: the function is simply
// called with two such objects denoting the values of the shape functions
// on the quadrature points of the two sides of the face. We then do what we
// always do: we fill the scratch arrays with the values of shape functions
// and their derivatives, and then loop over all entries of the matrix to
// compute the local integrals. The details of the bilinear form we evaluate
// here are given in the introduction.
template <int dim>
void
StokesDarcy<dim>::
assemble_interface_term (const FEFaceValuesBase<dim>          &darcy_fe_face_values,
		const FEFaceValuesBase<dim>          &stokes_fe_face_values,
		std::vector<double>                  &darcy_phi_p,
		std::vector<SymmetricTensor<2,dim> > &stokes_symgrad_phi_u,
		std::vector<Tensor<1,dim> >          &stokes_phi_u,
		std::vector<double>                  &stokes_phi_p,
		FullMatrix<double>                   &local_interface_matrix_velocity_pressure,
		FullMatrix<double>                   &local_interface_matrix_pressure_velocity,
		FullMatrix<double>                   &local_interface_matrix_velocity_velocity) const
{
	Assert (stokes_fe_face_values.n_quadrature_points ==
			darcy_fe_face_values.n_quadrature_points,
			ExcInternalError());
	const unsigned int n_face_quadrature_points
	= darcy_fe_face_values.n_quadrature_points;

	const FEValuesExtractors::Vector velocities (0);
	const FEValuesExtractors::Scalar pressure (dim);
	const FEValuesExtractors::Scalar darcy_pressure (dim+1);

	local_interface_matrix_velocity_pressure = 0;
	local_interface_matrix_pressure_velocity = 0;
	local_interface_matrix_velocity_velocity = 0;

	//std::cout << "stokes_fe_face_values.dofs_per_cell " << stokes_fe_face_values.dofs_per_cell << "\n";
	//std::cout << "darcy_fe_face_values.dofs_per_cell " << darcy_fe_face_values.dofs_per_cell << "\n";

	for (unsigned int q=0; q<n_face_quadrature_points; ++q)
	{
		const Tensor<1,dim> normal_vector = stokes_fe_face_values.normal_vector(q);
		for (unsigned int k=0; k<stokes_fe_face_values.dofs_per_cell; ++k)
		{
			stokes_symgrad_phi_u[k] = stokes_fe_face_values[velocities].symmetric_gradient (k, q);
			stokes_phi_u[k]= stokes_fe_face_values[velocities].value(k,q);
		}
		for (unsigned int k=0; k<darcy_fe_face_values.dofs_per_cell; ++k)
			darcy_phi_p[k] = darcy_fe_face_values[darcy_pressure].value (k,q);


		//assemble velocity velocity term
		switch(dim)
		{
		case 2:
		{
			Tensor<1,dim> tangent_vector;
			tangent_vector[0] = -1.0*normal_vector[1];
			tangent_vector[1] = normal_vector[0];

			for (unsigned int i=0; i<stokes_fe_face_values.dofs_per_cell; ++i)
				for (unsigned int j=0; j<stokes_fe_face_values.dofs_per_cell; ++j)
					local_interface_matrix_velocity_velocity(i,j) +=(1.0/G)*((stokes_phi_u[i]*tangent_vector)*(stokes_phi_u[j]*tangent_vector))*stokes_fe_face_values.JxW(q);
			break;
		}
		case 3 :
                			{
			Tensor<1,dim> tangent_vector;
			tangent_vector[0] = 1.0;
			tangent_vector[1] = 0.0;
			tangent_vector[2] = 0.0;
			for (unsigned int i=0; i<stokes_fe_face_values.dofs_per_cell; ++i)
				for (unsigned int j=0; j<stokes_fe_face_values.dofs_per_cell; ++j)
					local_interface_matrix_velocity_velocity(i,j) +=(1.0/G)*((stokes_phi_u[i]*tangent_vector)*(stokes_phi_u[j]*tangent_vector))*stokes_fe_face_values.JxW(q);
			break;
                			}
		}

		for (unsigned int i=0; i<darcy_fe_face_values.dofs_per_cell; ++i)
			for (unsigned int j=0; j<stokes_fe_face_values.dofs_per_cell; ++j)
			{
				local_interface_matrix_pressure_velocity(i,j) += (-(darcy_phi_p[i]*(stokes_phi_u[j]*normal_vector)))*stokes_fe_face_values.JxW(q);
				//std::cout << "stokes_fe_face_values.JxW(q) " << stokes_fe_face_values.JxW(q) << "\n";

			}
		for (unsigned int i=0; i<stokes_fe_face_values.dofs_per_cell; ++i)
			for (unsigned int j=0; j<darcy_fe_face_values.dofs_per_cell; ++j)
			{
				local_interface_matrix_velocity_pressure(i,j) += ((stokes_phi_u[i]*normal_vector)*(darcy_phi_p[j]))*stokes_fe_face_values.JxW(q);

			}
	}
}


template<int dim>
void
StokesDarcy<dim>::setup_darcy_mg_dofs()
{
	unsigned int n_levels = darcy_tri.n_levels();

	//Darcy objects for MG
	darcy_dof_handler.distribute_dofs(darcy_fe);
	darcy_dof_handler.distribute_mg_dofs(darcy_fe);

	std::vector<unsigned int> block_component(dim+2,1);
	block_component[dim]=2;
	block_component[dim+1]=0;

	//Loop through all levels and distribute dof's on each level
	for (unsigned int i=0; i << n_levels; ++i)
		DoFRenumbering::component_wise(darcy_dof_handler, i, block_component);

	std::cout << "Number of Darcy Dofs: " << darcy_dof_handler.n_dofs() << "\n";

	sparsity_pattern_darcy.reinit(darcy_dof_handler.n_dofs(),darcy_dof_handler.n_dofs(),
			darcy_dof_handler.max_couplings_between_dofs());

	DoFTools::make_sparsity_pattern(darcy_dof_handler,sparsity_pattern_darcy);

	mg_constraints_darcy.clear();
	DoFTools::make_hanging_node_constraints(darcy_dof_handler,mg_constraints_darcy);

	typename FunctionMap<dim>::type boundary_map;
	DarcyBoundaryValues<dim> darcy_boundary_function;
	boundary_map[1] = &darcy_boundary_function;

	const FEValuesExtractors::Scalar darcy_pressure(dim+1);
	VectorTools::interpolate_boundary_values(darcy_dof_handler,boundary_map,mg_constraints_darcy,
			darcy_fe.component_mask(darcy_pressure));

	mg_constraints_darcy.close();
	mg_constraints_darcy.condense (sparsity_pattern_darcy);
	sparsity_pattern_darcy.compress();

	darcy_mg_constrained_dofs.clear();
	darcy_mg_constrained_dofs.initialize(darcy_dof_handler,boundary_map);

	darcy_mg_interface_matrices.resize(0, n_levels-1);
	darcy_mg_interface_matrices.clear ();
	darcy_mg_matrices.resize(0, n_levels-1);
	darcy_mg_matrices.clear ();
	darcy_mg_sparsity_patterns.resize(0, n_levels-1);


    for (unsigned int level=0; level<n_levels; ++level)
    {
        CompressedSparsityPattern csp;
        csp.reinit(darcy_dof_handler.n_dofs(level),
                   darcy_dof_handler.n_dofs(level));
        MGTools::make_sparsity_pattern(darcy_dof_handler, csp, level);
        darcy_mg_sparsity_patterns[level].copy_from (csp);
        darcy_mg_matrices[level].reinit(darcy_mg_sparsity_patterns[level]);
        darcy_mg_interface_matrices[level].reinit(darcy_mg_sparsity_patterns[level]);
      }
}

template<int dim>
void StokesDarcy<dim>::setup_stokes_mg_dofs()
{
	// Stokes objects for MG
	unsigned int n_levels = stokes_tri.n_levels();

//	FESystem<dim> stokes_mg_fe (FE_Q<dim>(stokes_degree+1), dim, //Velocity vector
//					FE_Nothing<dim>(), 1,     //Pressure scalar zero extension since we only need (\grad u,\grad v)
//					FE_Nothing<dim>(), 1);    //Darcy pressure zero extension

	stokes_dof_handler.distribute_dofs(stokes_mg_fe);
	stokes_dof_handler.distribute_mg_dofs(stokes_mg_fe);

	std::vector<unsigned int> block_component(dim+2,1);
	block_component[dim]=2;
	block_component[dim+1]=0;

	//Loop through all levels and distribute dof's on each level
	for (unsigned int i=0; i << n_levels; ++i)
		DoFRenumbering::component_wise(stokes_dof_handler, i, block_component);

	std::cout << "Number of Stokes Dofs: " << stokes_dof_handler.n_dofs() << "\n";

	sparsity_pattern_stokes.reinit(stokes_dof_handler.n_dofs(),stokes_dof_handler.n_dofs(),
			stokes_dof_handler.max_couplings_between_dofs());

	DoFTools::make_sparsity_pattern(stokes_dof_handler,sparsity_pattern_stokes);

	mg_constraints_stokes.clear();
	DoFTools::make_hanging_node_constraints(stokes_dof_handler,mg_constraints_stokes);

	typename FunctionMap<dim>::type boundary_map;
	StokesBoundaryValues<dim> stokes_boundary_function;
	boundary_map[1] = &stokes_boundary_function;

	const FEValuesExtractors::Vector velocities(0);
	VectorTools::interpolate_boundary_values(stokes_dof_handler,boundary_map,mg_constraints_stokes,
			stokes_fe.component_mask(velocities));

	mg_constraints_stokes.close();
	mg_constraints_stokes.condense (sparsity_pattern_stokes);
	sparsity_pattern_stokes.compress();

	stokes_mg_constrained_dofs.clear();
	stokes_mg_constrained_dofs.initialize(stokes_dof_handler,boundary_map);

	stokes_mg_interface_matrices.resize(0, n_levels-1);
	stokes_mg_interface_matrices.clear ();
	stokes_mg_matrices.resize(0, n_levels-1);
	stokes_mg_matrices.clear ();
	stokes_mg_sparsity_patterns.resize(0, n_levels-1);


	for (unsigned int level=0; level<n_levels; ++level)
	{
		CompressedSparsityPattern csp;
		csp.reinit(stokes_dof_handler.n_dofs(level),
				stokes_dof_handler.n_dofs(level));
		MGTools::make_sparsity_pattern(stokes_dof_handler, csp, level);
		stokes_mg_sparsity_patterns[level].copy_from (csp);
		stokes_mg_matrices[level].reinit(stokes_mg_sparsity_patterns[level]);
		stokes_mg_interface_matrices[level].reinit(stokes_mg_sparsity_patterns[level]);
	}

}

template<int dim>
void
StokesDarcy<dim>::assemble_darcy_multigrid()
{
//	std::cout << darcy_tri.n_levels() << "\n";

	QGauss<dim>  quadrature_formula_darcy(1+darcy_degree);

	FEValues<dim> fe_values_darcy (darcy_fe, quadrature_formula_darcy,
			update_values   | update_gradients |
			update_quadrature_points | update_JxW_values);

	const unsigned int   dofs_per_cell_darcy   = darcy_fe.dofs_per_cell;
	const unsigned int   n_q_points_darcy      = quadrature_formula_darcy.size();

	std::vector<double>         darcy_phi_p          (dofs_per_cell_darcy);  //Darcy pressure values
	std::vector<Tensor<1,dim>>  darcy_grad_phi       (dofs_per_cell_darcy);  //Darcy gradient p values

	FullMatrix<double>   cell_matrix_darcy (dofs_per_cell_darcy, dofs_per_cell_darcy);
	std::vector<types::global_dof_index> local_dof_indices_darcy (dofs_per_cell_darcy);

	std::vector<std::vector<bool> > interface_dofs_darcy
	= darcy_mg_constrained_dofs.get_refinement_edge_indices ();

	std::vector<std::vector<bool> > boundary_interface_dofs_darcy
	= darcy_mg_constrained_dofs.get_refinement_edge_boundary_indices ();

	std::vector<ConstraintMatrix> boundary_constraints_darcy (darcy_tri.n_levels());
	std::vector<ConstraintMatrix> boundary_interface_constraints_darcy (darcy_tri.n_levels());

	for (unsigned int level=0; level<darcy_tri.n_levels(); ++level)
	{
		boundary_constraints_darcy[level].add_lines (interface_dofs_darcy[level]);
		boundary_constraints_darcy[level].add_lines (darcy_mg_constrained_dofs.get_boundary_indices()[level]);
		boundary_constraints_darcy[level].close ();

		boundary_interface_constraints_darcy[level].add_lines (boundary_interface_dofs_darcy[level]);
		boundary_interface_constraints_darcy[level].close ();
	}


	for (unsigned int q=0; q<fe_values_darcy.n_quadrature_points; ++q)
	{
		for (unsigned int k=0; k<dofs_per_cell_darcy; ++k)
		{
			darcy_grad_phi[k] = fe_values_darcy.shape_grad (k, q);
			//darcy_phi_p[k] = fe_values_darcy.shape_value(k,q);
		}

	typename DoFHandler<dim>::cell_iterator cell = darcy_dof_handler.begin(),
			endc = darcy_dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		cell_matrix_darcy = 0;
		fe_values_darcy.reinit (cell);

		//Modify so that its our darcy
	for (unsigned int i=0; i<dofs_per_cell_darcy; ++i)
		for (unsigned int j=0; j<dofs_per_cell_darcy; ++j)
				cell_matrix_darcy(i,j) += ((darcy_grad_phi[i]*darcy_grad_phi[j])*fe_values_darcy.JxW(q));

	cell->get_mg_dof_indices (local_dof_indices_darcy);


	boundary_constraints_darcy[cell->level()].distribute_local_to_global (cell_matrix_darcy,
		  local_dof_indices_darcy,darcy_mg_matrices[cell->level()]);

		for (unsigned int i=0; i<dofs_per_cell_darcy; ++i)
			for (unsigned int j=0; j<dofs_per_cell_darcy; ++j)
				if ( !(interface_dofs_darcy[cell->level()][local_dof_indices_darcy[i]]==true &&
						interface_dofs_darcy[cell->level()][local_dof_indices_darcy[j]]==false))
					cell_matrix_darcy(i,j) = 0;

		boundary_interface_constraints_darcy[cell->level()]
		                               .distribute_local_to_global (cell_matrix_darcy,
		                            		   local_dof_indices_darcy,
		                            		   darcy_mg_interface_matrices[cell->level()]);
	}

	}

}

template<int dim>
void StokesDarcy<dim>::assemble_stokes_multigrid()
{
	const QGauss<dim> stokes_quadrature(stokes_degree+2);

	FEValues<dim> fe_values (stokes_mg_fe, stokes_quadrature,
	                           update_values    |
	                           update_quadrature_points  |
	                           update_JxW_values |
	                           update_gradients);

	const unsigned int	stokes_dofs_per_cell = stokes_mg_fe.dofs_per_cell;
	const unsigned int  n_q_points_stokes = stokes_quadrature.size();

	const FEValuesExtractors::Vector     velocities (0);

	std::vector<SymmetricTensor<2,dim> > stokes_symgrad_phi_u (stokes_dofs_per_cell);

	FullMatrix<double>   cell_matrix_stokes (stokes_dofs_per_cell, stokes_dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices_stokes (stokes_dofs_per_cell);

	std::vector<std::vector<bool> > interface_dofs_stokes
	= stokes_mg_constrained_dofs.get_refinement_edge_indices ();

	std::vector<std::vector<bool> > boundary_interface_dofs_stokes
	= stokes_mg_constrained_dofs.get_refinement_edge_boundary_indices ();

	std::vector<ConstraintMatrix> boundary_constraints_stokes (stokes_tri.n_levels());
	std::vector<ConstraintMatrix> boundary_interface_constraints_stokes (stokes_tri.n_levels());

	for (unsigned int level=0; level<darcy_tri.n_levels(); ++level)
	{
		boundary_constraints_stokes[level].add_lines (interface_dofs_stokes[level]);
		boundary_constraints_stokes[level].add_lines (darcy_mg_constrained_dofs.get_boundary_indices()[level]);
		boundary_constraints_stokes[level].close ();

		boundary_interface_constraints_stokes[level].add_lines (boundary_interface_dofs_stokes[level]);
		boundary_interface_constraints_stokes[level].close ();
	}

	for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
		{
			for (unsigned int k=0; k<stokes_dofs_per_cell; ++k)
			{
				stokes_symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);;

			}

		typename DoFHandler<dim>::cell_iterator cell = stokes_dof_handler.begin(),
				endc = stokes_dof_handler.end();
		for (; cell!=endc; ++cell)
		{
			cell_matrix_stokes = 0;
			fe_values.reinit (cell);


		for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
			for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
				cell_matrix_stokes(i,j) += (2 * viscosity * stokes_symgrad_phi_u[i] * stokes_symgrad_phi_u[j])* fe_values.JxW(q);

		cell->get_mg_dof_indices (local_dof_indices_stokes);


		boundary_constraints_stokes[cell->level()].distribute_local_to_global (cell_matrix_stokes,
				local_dof_indices_stokes,stokes_mg_matrices[cell->level()]);

			for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
				for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
					if ( !(interface_dofs_stokes[cell->level()][local_dof_indices_stokes[i]]==true &&
							interface_dofs_stokes[cell->level()][local_dof_indices_stokes[j]]==false))
						cell_matrix_stokes(i,j) = 0;

			boundary_interface_constraints_stokes[cell->level()]
			                               .distribute_local_to_global (cell_matrix_stokes,
			                            		   local_dof_indices_stokes,
			                            		   stokes_mg_interface_matrices[cell->level()]);
		}

		}


}


//Method to construct the Multigrid preconditioner
template<int dim>
void StokesDarcy<dim>::construct_darcy_preconditioner()
{
	MGTransferPrebuilt<Vector<double> > mg_transfer_darcy(mg_constraints_darcy, darcy_mg_constrained_dofs);

	mg_transfer_darcy.build_matrices(darcy_dof_handler);
	FullMatrix<double> coarse_matrix_darcy;
	coarse_matrix_darcy.copy_from (darcy_mg_matrices[0]);
	MGCoarseGridHouseholder<> coarse_grid_solver_darcy;
	coarse_grid_solver_darcy.initialize (coarse_matrix_darcy);

	typedef PreconditionSOR<SparseMatrix<double> > Smoother;
	mg::SmootherRelaxation<Smoother, Vector<double> > mg_smoother_darcy;
	mg_smoother_darcy.initialize(darcy_mg_matrices);
	mg_smoother_darcy.set_steps(2);
	mg_smoother_darcy.set_symmetric(true);

	mg::Matrix<Vector<double>> mg_matrix_darcy(darcy_mg_matrices);
	mg::Matrix<Vector<double>> mg_interface_up_darcy(darcy_mg_interface_matrices);
	mg::Matrix<Vector<double>> mg_interface_down_darcy(darcy_mg_interface_matrices);

	Multigrid<Vector<double> > mg_darcy(darcy_dof_handler,
			mg_matrix_darcy,
			coarse_grid_solver_darcy,
			mg_transfer_darcy,
			mg_smoother_darcy,
			mg_smoother_darcy);

	mg_darcy.set_edge_matrices(mg_interface_down_darcy, mg_interface_up_darcy);
	PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>
	darcy_preconditioner(darcy_dof_handler, mg_darcy, mg_transfer_darcy);
}

template<int dim>
void StokesDarcy<dim>::construct_stokes_preconditioner()
{
	MGTransferPrebuilt<Vector<double> > mg_transfer_stokes(mg_constraints_stokes, stokes_mg_constrained_dofs);

	mg_transfer_stokes.build_matrices(stokes_dof_handler);
	FullMatrix<double> coarse_matrix_stokes;
	coarse_matrix_stokes.copy_from (stokes_mg_matrices[0]);
	MGCoarseGridHouseholder<> coarse_grid_solver_stokes;
	coarse_grid_solver_stokes.initialize (coarse_matrix_stokes);

	typedef PreconditionSOR<SparseMatrix<double> > Smoother;
	mg::SmootherRelaxation<Smoother, Vector<double> > mg_smoother_stokes;
	mg_smoother_stokes.initialize(stokes_mg_matrices);
	mg_smoother_stokes.set_steps(2);
	mg_smoother_stokes.set_symmetric(true);

	mg::Matrix<Vector<double>> mg_matrix_stokes(stokes_mg_matrices);
	mg::Matrix<Vector<double>> mg_interface_up_stokes(stokes_mg_interface_matrices);
	mg::Matrix<Vector<double>> mg_interface_down_stokes(stokes_mg_interface_matrices);

	Multigrid<Vector<double> > mg_stokes(stokes_dof_handler,
			mg_matrix_stokes,
			coarse_grid_solver_stokes,
			mg_transfer_stokes,
			mg_smoother_stokes,
			mg_smoother_stokes);

	mg_stokes.set_edge_matrices(mg_interface_down_stokes, mg_interface_up_stokes);
	PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>
	stokes_preconditioner(stokes_dof_handler, mg_stokes, mg_transfer_stokes);
}



// @sect4{<code>StokesDarcy::solve</code>}

// As discussed in the introduction, we use a rather trivial solver here: we
// just pass the linear system off to the SparseDirectUMFPACK direct solver
// (see, for example, step-29). The only thing we have to do after solving
// is ensure that hanging node and boundary value constraints are correct.
template <int dim>
void
StokesDarcy<dim>::solve (const unsigned int refinement_cycle, const unsigned int solution_type, const unsigned int block_pattern)
{
	switch (solution_type)
	{
	case 0 :
	{
		switch (block_pattern)
		{
		case 2 :
		{

			SparseMatrix<double> block_diagonal(sparsity_pattern.block(0,0));
			block_diagonal.copy_from(system_matrix.block(0,0));
			block_diagonal.symmetrize();

			FullMatrix<double> identity (IdentityMatrix(solution.block(1).size()));

			SparseMatrix<double> scaled_identity(sparsity_pattern.block(1,1));
			scaled_identity.copy_from(identity);
			scaled_identity *= 0.6;

			InverseMatrix<SparseMatrix<double>> a_preconditioner(block_diagonal);
			InverseMatrix<SparseMatrix<double>> schur_preconditioner(scaled_identity);

//			SchurComplement schur_complement(system_matrix,a_preconditioner);
//
//			PreconditionIdentity schur_identity;
//			IterativeInverse<Vector<double>> schur_solver;
//			schur_solver.initialize(schur_complement,schur_identity);
//			schur_solver.solver.select("minres");
//			static ReductionControl inner_control(1000,1e-6);
//			schur_solver.solver.set_control(inner_control);
//
//			BlockSchurPreconditioner<SparseMatrix<double>> constraint_preconditioner(system_matrix,a_preconditioner,schur_solver);
			BlockPreconditioner<SparseMatrix<double>> preconditioner(system_matrix,a_preconditioner,schur_preconditioner,pre_type, block_pattern);

			// ***Make tolerance equal tol*norm(r0) so that it matches with matlab stopping criterion***

			SolverControl solver_control(system_matrix.m(),1e-8*system_rhs.l2_norm());
			solver_control.enable_history_data();
			//solver_control.log_history(true);

			SolverGMRES<BlockVector<double> >::AdditionalData gmres_data;
			GrowingVectorMemory<BlockVector<double> > vector_memory;
			gmres_data.max_n_tmp_vectors = system_matrix.m();
			gmres_data.right_preconditioning = true;
			//gmres_data.use_default_residual = false;

			SolverGMRES<BlockVector<double>> solver (solver_control,vector_memory,gmres_data);

			Timer time;
			time.start();
			solver.solve (system_matrix, solution, system_rhs, preconditioner);
			time.stop();
			constraints.distribute (solution);

			std::vector<double> residuals(solver_control.last_step()+1);
			solver_control.residual_out(solver_control.last_step()+1,residuals);

			std::ostringstream res_name;
			res_name << "residual_vector_cycle" << Utilities::int_to_string(refinement_cycle,2) <<".m";

			std::ofstream res_out(res_name.str().c_str());
			std::ostream_iterator<double> output_iterator(res_out,"\n");
			std::copy(residuals.begin(), residuals.end(), output_iterator);

			iteration_table.add_value("cycle", refinement_cycle);
			iteration_table.add_value("cells", triangulation.n_active_cells());
			iteration_table.add_value("dofs",dof_handler.n_dofs());
			iteration_table.add_value("GMRES iterations", solver_control.last_step());
			iteration_table.add_value("CPU Time",time());
			break;
		}
		case 3:
		{
			//SETUP DARCY MG PRECONDITIONER
			MGTransferPrebuilt<Vector<double> > mg_transfer_darcy(mg_constraints_darcy, darcy_mg_constrained_dofs);

			mg_transfer_darcy.build_matrices(darcy_dof_handler);
			FullMatrix<double> coarse_matrix_darcy;
			coarse_matrix_darcy.copy_from (darcy_mg_matrices[0]);
			MGCoarseGridHouseholder<> coarse_grid_solver_darcy;
			coarse_grid_solver_darcy.initialize (coarse_matrix_darcy);

			typedef PreconditionSOR<SparseMatrix<double> > Smoother;
			mg::SmootherRelaxation<Smoother, Vector<double> > mg_smoother_darcy;
			mg_smoother_darcy.initialize(darcy_mg_matrices);
			mg_smoother_darcy.set_steps(10);
			mg_smoother_darcy.set_symmetric(false);

			mg::Matrix<Vector<double>> mg_matrix_darcy(darcy_mg_matrices);
			mg::Matrix<Vector<double>> mg_interface_up_darcy(darcy_mg_interface_matrices);
			mg::Matrix<Vector<double>> mg_interface_down_darcy(darcy_mg_interface_matrices);

			Multigrid<Vector<double> > mg_darcy(darcy_dof_handler,
					mg_matrix_darcy,
					coarse_grid_solver_darcy,
					mg_transfer_darcy,
					mg_smoother_darcy,
					mg_smoother_darcy);

			mg_darcy.set_edge_matrices(mg_interface_down_darcy, mg_interface_up_darcy);
			PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>
			darcy_preconditioner(darcy_dof_handler, mg_darcy, mg_transfer_darcy);

			//SETUP STOKES MG PRECONDITIONER
			MGTransferPrebuilt<Vector<double> > mg_transfer_stokes(mg_constraints_stokes, stokes_mg_constrained_dofs);

			mg_transfer_stokes.build_matrices(stokes_dof_handler);
			FullMatrix<double> coarse_matrix_stokes;
			coarse_matrix_stokes.copy_from (stokes_mg_matrices[0]);
			MGCoarseGridHouseholder<> coarse_grid_solver_stokes;
			coarse_grid_solver_stokes.initialize (coarse_matrix_stokes);

			typedef PreconditionSOR<SparseMatrix<double> > Smoother;
			mg::SmootherRelaxation<Smoother, Vector<double> > mg_smoother_stokes;
			mg_smoother_stokes.initialize(stokes_mg_matrices);
			mg_smoother_stokes.set_steps(10);
			mg_smoother_stokes.set_symmetric(false);

			mg::Matrix<Vector<double>> mg_matrix_stokes(stokes_mg_matrices);
			mg::Matrix<Vector<double>> mg_interface_up_stokes(stokes_mg_interface_matrices);
			mg::Matrix<Vector<double>> mg_interface_down_stokes(stokes_mg_interface_matrices);

			Multigrid<Vector<double> > mg_stokes(stokes_dof_handler,
					mg_matrix_stokes,
					coarse_grid_solver_stokes,
					mg_transfer_stokes,
					mg_smoother_stokes,
					mg_smoother_stokes);

			mg_stokes.set_edge_matrices(mg_interface_down_stokes, mg_interface_up_stokes);
			PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>
			stokes_preconditioner(stokes_dof_handler, mg_stokes, mg_transfer_stokes);

			SparseMatrix<double> P1(sparsity_pattern.block(0,0));
			P1.copy_from(system_matrix.block(0,0));

			SparseMatrix<double> P2(sparsity_pattern.block(1,1));
			P2.copy_from(system_matrix.block(1,1));

//			FullMatrix<double> identity (IdentityMatrix(solution.block(2).size()));

			SparseMatrix<double> scaled_identity(sparsity_pattern.block(2,2),IdentityMatrix(solution.block(2).size()));
//			scaled_identity.copy_from(identity);
			scaled_identity *= 0.6;

			InverseMatrix<SparseMatrix<double>> a_preconditioner(P1);
			InverseMatrix<SparseMatrix<double>> b_preconditioner(P2);
			InverseMatrix<SparseMatrix<double>> schur_preconditioner(scaled_identity);

			BlockPreconditioner<SparseMatrix<double>> preconditioner(system_matrix,a_preconditioner,b_preconditioner,schur_preconditioner,pre_type, block_pattern);


//			SchurComplement<SparseMatrix<double>,PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>>
			SchurComplement<InverseMatrix<SparseMatrix<double>>>
			schur_complement(system_matrix,b_preconditioner,3);

//			IdentityMatrix schur_identity(solution.block(2).size());
			IterativeInverse<Vector<double>> schur_solver;
			schur_solver.initialize(schur_complement,scaled_identity);
			schur_solver.solver.select("minres");
			static ReductionControl inner_control(1000,1e-6);
			schur_solver.solver.set_control(inner_control);


			Block_MG_Preconditioner<PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>,
			PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>,
			InverseMatrix<SparseMatrix<double>>> preconditioner_mg_bd
			(system_matrix,darcy_preconditioner,stokes_preconditioner,schur_preconditioner,1,block_pattern);

			Block_MG_Preconditioner<PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>,
			PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>,
			InverseMatrix<SparseMatrix<double>>> preconditioner_mg_lt
			(system_matrix,darcy_preconditioner,stokes_preconditioner,schur_preconditioner,2,block_pattern);

			Block_MG_Preconditioner<PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>,
			PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>,
			IterativeInverse<Vector<double>>> preconditioner_mg_con
			(system_matrix,darcy_preconditioner,stokes_preconditioner,schur_solver,pre_type,block_pattern);

			Block_MG_Preconditioner<InverseMatrix<SparseMatrix<double>>,
			InverseMatrix<SparseMatrix<double>>,
			IterativeInverse<Vector<double>>> preconditioner_mg_exact
			(system_matrix,a_preconditioner,b_preconditioner,schur_solver,pre_type,block_pattern);


			// ***Make tolerance equal tol*norm(r0) so that it matches with matlab stopping criterion***

			SolverControl solver_control(system_matrix.m(),1e-8*system_rhs.l2_norm());
			solver_control.enable_history_data();
			//solver_control.log_history(true);

			SolverGMRES<BlockVector<double> >::AdditionalData gmres_data;
			GrowingVectorMemory<BlockVector<double> > vector_memory;
			gmres_data.max_n_tmp_vectors = system_matrix.m();
			gmres_data.right_preconditioning = true;
			//gmres_data.use_default_residual = false;

			SolverGMRES<BlockVector<double>> solver (solver_control,vector_memory,gmres_data);

//			SolverFGMRES<BlockVector<double>>::AdditionalData fgmres_data;
//			SolverFGMRES<BlockVector<double>> solver_test(solver_control, vector_memory,fgmres_data);

			Timer time;
			time.start();
			solver.solve (system_matrix, solution, system_rhs, preconditioner_mg_bd);
			time.stop();

			schur_solver.clear();
			constraints.distribute (solution);

			std::vector<double> residuals(solver_control.last_step()+1);
			solver_control.residual_out(solver_control.last_step()+1,residuals);

			std::ostringstream res_name;
			res_name << "residual_vector_cycle" << Utilities::int_to_string(refinement_cycle,2) <<".m";

			std::ofstream res_out(res_name.str().c_str());
			std::ostream_iterator<double> output_iterator(res_out,"\n");
			std::copy(residuals.begin(), residuals.end(), output_iterator);

			iteration_table.add_value("cycle", refinement_cycle);
			iteration_table.add_value("cells", triangulation.n_active_cells());
			iteration_table.add_value("dofs",dof_handler.n_dofs());
			iteration_table.add_value("GMRES iterations", solver_control.last_step());
			iteration_table.add_value("CPU Time",time());
			break;
		}

		}

	break ;
	}
	case 1 :
	{
		SparseDirectUMFPACK direct_solver;
		direct_solver.initialize (system_matrix);
		direct_solver.vmult (solution, system_rhs);
		break;
	}

	default :
	{
		Assert (false, ExcNotImplemented());
	}

	}

	/*
   std::ostringstream matname;
   matname << "sysmatrix-dim" << Utilities::int_to_string(dim,1)
               << Utilities::int_to_string (refinement_cycle, 2)
               << ".m";

   std::ofstream matout (matname.str().c_str());
   system_matrix.print(matout);

   std::ostringstream rhsname;
   rhsname << "rhs-dim" << Utilities::int_to_string(dim,1)
   	   	   << Utilities::int_to_string (refinement_cycle, 2)
   	   	   << ".m";

   std::ofstream rhsout(rhsname.str().c_str());
   system_rhs.print(rhsout);
	 */
	constraints.distribute (solution);
	system_matrix.clear();
}




// @sect4{<code>StokesDarcy::output_results</code>}

// Generating graphical output is rather trivial here: all we have to do is
// identify which components of the solution vector belong to scalars and/or
// vectors (see, for example, step-22 for a previous example), and then pass
// it all on to the DataOut class (with the second template argument equal
// to hp::DoFHandler instead of the usual default DoFHandler):
template <int dim>
void
StokesDarcy<dim>::
output_results (const unsigned int refinement_cycle)  const
{
	std::vector<std::string> solution_names (dim, "velocity");
	solution_names.push_back ("pressure");
	solution_names.push_back ("darcy_pressure");

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	data_component_interpretation
	(dim, DataComponentInterpretation::component_is_part_of_vector);
	data_component_interpretation
	.push_back (DataComponentInterpretation::component_is_scalar);
	data_component_interpretation
	.push_back (DataComponentInterpretation::component_is_scalar);

	DataOut<dim,hp::DoFHandler<dim> > data_out;
	data_out.attach_dof_handler (dof_handler);

	data_out.add_data_vector (solution, solution_names,
			DataOut<dim,hp::DoFHandler<dim> >::type_dof_data,
			data_component_interpretation);
	data_out.build_patches ();

	std::ostringstream filename;
	filename << "solution-" << Utilities::int_to_string(dim,1)
	<< Utilities::int_to_string (refinement_cycle, 2)
	<< ".vtk";

	std::ofstream output (filename.str().c_str());
	data_out.write_vtk (output);

}


// @sect4{<code>StokesDarcy::refine_mesh</code>}

// The next step is to refine the mesh. As was discussed in the
// introduction, this is a bit tricky primarily because the fluid and the
// darcy subdomains use variables that have different physical dimensions
// and for which the absolute magnitude of error estimates is consequently
// not directly comparable. We will therefore have to scale them. At the top
// of the function, we therefore first compute error estimates for the
// different variables separately (using the velocities but not the pressure
// for the fluid domain, and the darcy_pressure in the darcy domain):
template <int dim>
void
StokesDarcy<dim>::refine_mesh ()
{
	Vector<float>
	stokes_estimated_error_per_cell (triangulation.n_active_cells());
	Vector<float>
	darcy_estimated_error_per_cell (triangulation.n_active_cells());

	const QGauss<dim-1> stokes_face_quadrature(stokes_degree+2);
	const QGauss<dim-1> darcy_face_quadrature(darcy_degree+2);

	hp::QCollection<dim-1> face_q_collection;
	face_q_collection.push_back (stokes_face_quadrature);
	face_q_collection.push_back (darcy_face_quadrature);

	const FEValuesExtractors::Vector velocities(0);
	KellyErrorEstimator<dim>::estimate (dof_handler,
			face_q_collection,
			typename FunctionMap<dim>::type(),
			solution,
			stokes_estimated_error_per_cell,
			fe_collection.component_mask(velocities));

	const FEValuesExtractors::Scalar darcy_pressure(dim); //darcy pressure scalar
	KellyErrorEstimator<dim>::estimate (dof_handler,
			face_q_collection,
			typename FunctionMap<dim>::type(),
			solution,
			darcy_estimated_error_per_cell,
			fe_collection.component_mask(darcy_pressure));

	// We then normalize error estimates by dividing by their norm and scale
	// the fluid error indicators by a factor of 4 as discussed in the
	// introduction. The results are then added together into a vector that
	// contains error indicators for all cells:
	stokes_estimated_error_per_cell
	*= 4. / stokes_estimated_error_per_cell.l2_norm();
	darcy_estimated_error_per_cell
	*= 1. / darcy_estimated_error_per_cell.l2_norm();

	Vector<float>
	estimated_error_per_cell (triangulation.n_active_cells());

	estimated_error_per_cell += stokes_estimated_error_per_cell;
	estimated_error_per_cell += darcy_estimated_error_per_cell;

	// The second to last part of the function, before actually refining the
	// mesh, involves a heuristic that we have already mentioned in the
	// introduction: because the solution is discontinuous, the
	// KellyErrorEstimator class gets all confused about cells that sit at the
	// boundary between subdomains: it believes that the error is large there
	// because the jump in the gradient is large, even though this is entirely
	// expected and a feature that is in fact present in the exact solution as
	// well and therefore not indicative of any numerical error.
	//
	// Consequently, we set the error indicators to zero for all cells at the
	// interface; the conditions determining which cells this affects are
	// slightly awkward because we have to account for the possibility of
	// adaptively refined meshes, meaning that the neighboring cell can be
	// coarser than the current one, or could in fact be refined some
	// more. The structure of these nested conditions is much the same as we
	// encountered when assembling interface terms in
	// <code>assemble_system</code>.
	{
		unsigned int cell_index = 0;
		for (typename hp::DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell, ++cell_index)
			for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
				if (cell_is_in_darcy_domain (cell))
				{
					if ((cell->at_boundary(f) == false)
							&&
							(((cell->neighbor(f)->level() == cell->level())
									&&
									(cell->neighbor(f)->has_children() == false)
									&&
									cell_is_in_fluid_domain (cell->neighbor(f)))
									||
									((cell->neighbor(f)->level() == cell->level())
											&&
											(cell->neighbor(f)->has_children() == true)
											&&
											(cell_is_in_fluid_domain (cell->neighbor_child_on_subface
													(f, 0))))
													||
													(cell->neighbor_is_coarser(f)
															&&
															cell_is_in_fluid_domain(cell->neighbor(f)))
							))
						estimated_error_per_cell(cell_index) = 0;
				}
				else
				{
					if ((cell->at_boundary(f) == false)
							&&
							(((cell->neighbor(f)->level() == cell->level())
									&&
									(cell->neighbor(f)->has_children() == false)
									&&
									cell_is_in_darcy_domain (cell->neighbor(f)))
									||
									((cell->neighbor(f)->level() == cell->level())
											&&
											(cell->neighbor(f)->has_children() == true)
											&&
											(cell_is_in_darcy_domain (cell->neighbor_child_on_subface
													(f, 0))))
													||
													(cell->neighbor_is_coarser(f)
															&&
															cell_is_in_darcy_domain(cell->neighbor(f)))
							))
						estimated_error_per_cell(cell_index) = 0;
				}
	}

	GridRefinement::refine_and_coarsen_fixed_number (triangulation,
			estimated_error_per_cell,
			0.3, 0.0);
	triangulation.execute_coarsening_and_refinement ();
}

template <int dim>
void StokesDarcy<dim>::compute_errors(const unsigned int refinement_cycle)
{

	const FEValuesExtractors::Vector velocities (0);
	const FEValuesExtractors::Scalar pressure (dim);
	const FEValuesExtractors::Scalar darcy_pressure (dim+1);


	const ComponentSelectFunction<dim>
	darcy_pressure_mask (dim+1, dim+2);

	const ComponentSelectFunction<dim>
	pressure_mask (dim, dim+2);

	const ComponentSelectFunction<dim>
	velocity_mask(std::make_pair(0, dim), dim+2);

	const QGauss<dim> stokes_quadrature(stokes_degree+2);
	const QGauss<dim> darcy_quadrature(darcy_degree+1);

	unsigned int cell_index = 0;

	hp::QCollection<dim>  q_collection;
	q_collection.push_back (stokes_quadrature);
	q_collection.push_back (darcy_quadrature);

	ExactSolution<dim> exact_solution;
	Vector<double> cellwise_errors_sv (triangulation.n_active_cells());
	Vector<double> cellwise_errors_sp (triangulation.n_active_cells());
	Vector<double> cellwise_errors_dp (triangulation.n_active_cells());
	//std::cout << "n active cells " << triangulation.n_active_cells() << std::endl;


	QTrapez<1>     q_trapez;
	QIterated<dim> quadrature (q_trapez, stokes_degree+2);

	VectorTools::integrate_difference(dof_handler,
			solution,
			exact_solution,
			cellwise_errors_sv,
			q_collection,
			VectorTools::L2_norm,
			&velocity_mask);


	for (typename hp::DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active();
			cell != dof_handler.end(); ++cell, ++cell_index)
	{
		if (cell_is_in_darcy_domain (cell))
		{
			cellwise_errors_sv(cell_index) =0.0;
		}
	}

	const double u_l2_error = cellwise_errors_sv.l2_norm();

	std::cout << "Stokes velocity L2 error = " << u_l2_error << std::endl;

	VectorTools::integrate_difference (dof_handler,
			solution,
			exact_solution,
			cellwise_errors_sp,
			q_collection,
			VectorTools::L2_norm,
			&pressure_mask);

	cell_index=0;
	for (typename hp::DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active();
			cell != dof_handler.end(); ++cell, ++cell_index)
	{
		if (cell_is_in_darcy_domain (cell))
		{
			cellwise_errors_sp(cell_index) =0.0;
		}
	}


	const double p_l2_error = cellwise_errors_sp.l2_norm();

	std::cout << "Stokes pressure L2  error = " << p_l2_error << std::endl;


	VectorTools::integrate_difference (dof_handler,solution, exact_solution,
			cellwise_errors_dp, q_collection,
			VectorTools::L2_norm,
			&darcy_pressure_mask);


	cell_index=0;
	for (typename hp::DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active();
			cell != dof_handler.end(); ++cell, ++cell_index)
	{
		if (cell_is_in_fluid_domain (cell))
		{
			cellwise_errors_dp(cell_index) =0.0;
		}
	}


	const double dp_l2_error = cellwise_errors_dp.l2_norm();

	std::cout << "Darcy pressure L2  error = " << dp_l2_error << std::endl;

	std::vector<unsigned int> block_component(dim+2,1);
	block_component[dim]=2;
	block_component[dim+1]=0;

	std::vector<types::global_dof_index> dofs_per_block (3);
	DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
	const unsigned int n_uf = dofs_per_block[1],
			n_pf = dofs_per_block[2],
			n_pd = dofs_per_block[0];

	convergence_table.add_value("cycle", refinement_cycle);
	convergence_table.add_value("cells", triangulation.n_active_cells());

	convergence_table.add_value("S_v dofs",n_uf);
	convergence_table.add_value("S_v error",u_l2_error);

	convergence_table.add_value("S_p dofs", n_pf);
	convergence_table.add_value("S_p error",p_l2_error);

	convergence_table.add_value("D_p dofs",n_pd);
	convergence_table.add_value("D_p error",dp_l2_error);


	/*VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::L2_norm,
                                        fe_collection.component_mask(pressure));
      const double p_l2_error = cellwise_errors.l2_norm();

      std::cout << "Stokes pressure L2 error = " << p_l2_error << std::endl;

      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_errors, quadrature,
                                         VectorTools::L2_norm,
                                         fe_collection.component_mask(darcy_pressure));
      const double dp_l2_error = cellwise_errors.l2_norm();

      std::cout << "Darcy pressure L2 error = " << dp_l2_error << std::endl;*/

	/*VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::H1_seminorm,
                                        fe_collection.component_mask(darcy_pressure));
      const double pd_h1_error = cellwise_errors.l2_norm();
      std::cout << "Darcy pressre  H1 error = " << dp_h1_error << std::endl;*/

}



// @sect4{<code>StokesDarcy::run</code>}

// This is, as usual, the function that controls the overall flow of
// operation. If you've read through tutorial programs step-1 through
// step-6, for example, then you are already quite familiar with the
// following structure:
template <int dim>
void StokesDarcy<dim>::run ()
{
	// Change the type of solution, type=0 - iterative; type=1 - direct
	unsigned int solution_type = 0;
	// Change the block pattern type, 2 means 2 x 2, 3 means 3 x 3, all others give error
	unsigned int block_pattern = 3;

	for (unsigned int refinement_cycle = 0; refinement_cycle<4;//0-2*dim;
			++refinement_cycle)
	{
		std::cout << "Refinement cycle " << refinement_cycle << std::endl;

		if (refinement_cycle == 0)
		{
			make_grid ();
		}
		else
		{
			triangulation.refine_global (1);
			stokes_tri.refine_global (1);
			darcy_tri.refine_global (1);
		}
		setup_dofs (block_pattern);

		std::cout << "   Assembling..." << std::endl;
		assemble_system ();

		std::cout << "Setting up Darcy MG dofs..." << std::endl;
		setup_darcy_mg_dofs();

		std::cout << "Setting up Stokes MG dofs..."<<std::endl;
		setup_stokes_mg_dofs();

		std::cout << "Assembling Darcy MG operator..." << std::endl;
		assemble_darcy_multigrid ();

		std::cout << "Assembling Stokes MG operator..." << std::endl;
		assemble_stokes_multigrid ();

		std::cout << "Number of degrees of freedom: "
				          << dof_handler.n_dofs()
				          << " (by level: ";
				  for (unsigned int level=0; level<triangulation.n_levels(); ++level)
				    std::cout << stokes_dof_handler.n_dofs(level)+ darcy_dof_handler.n_dofs(level)
				    << (level == triangulation.n_levels()-1 ? ")" : ", ") ;
				  std::cout << "\n";

		std::cout << "Solving..." << std::endl;

		solve (refinement_cycle,solution_type,block_pattern);
		compute_errors(refinement_cycle);

		std::cout << "Writing output..." << std::endl;
		output_results (refinement_cycle);

		std::cout << std::endl;
	}

    convergence_table.set_precision("S_v error", 3);
    convergence_table.set_precision("S_p error", 3);
    convergence_table.set_precision("D_p error", 3);
    convergence_table.set_scientific("S_v error", true);
    convergence_table.set_scientific("S_p error", true);
    convergence_table.set_scientific("D_p error", true);

    convergence_table.evaluate_convergence_rates("S_v error", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("S_p error", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("D_p error", ConvergenceTable::reduction_rate_log2);

	convergence_table.write_text(std::cout);
	std::cout<<std::endl;
	if (solution_type==0)
	iteration_table.write_text(std::cout);
}

//Ends CoupledProblem Namespace
}



// @sect4{The <code>main()</code> function}

// This, final, function contains pretty much exactly what most of the other
// tutorial programs have:
int main ()
{
	try
	{
		using namespace dealii;

		using namespace CoupledProblem;

		deallog.depth_console (3);
		int pre=1;

		StokesDarcy<2> flow_problem(1, 1, pre);
		flow_problem.run ();

//
//      StokesDarcy<3> flow_3D(1,1,1);
//      flow_3D.run();

	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Exception on processing: " << std::endl
				<< exc.what() << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Unknown exception!" << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
