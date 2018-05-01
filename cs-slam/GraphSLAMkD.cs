using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Generic;

namespace SLAM
{
    /// <summary>
    /// SLAM: Simultaneous Localization And Mapping
    /// Objective: this class allows points and landmarks to be multi-dimensional (cartesian-type space)
    /// </summary>
    public class GraphSLAMkD
    {
        Matrix mOmega;
        Matrix mXi;
        Dictionary<int, int[]> mMatrixIndexTable = new Dictionary<int, int[]>();
        int mDimension;

        public GraphSLAMkD(int dimension)
        {
            mDimension = dimension;    
        }
        
        public void AddStartingPointInfo(int starting_point_id, double[] starting_point_value)
        {
            if (mOmega == null)
            {
                mOmega = DenseMatrix.Identity(mDimension);

                int[] point_matrix_index_list = new int[mDimension];
                for (int i = 0; i < mDimension; ++i)
                {
                    point_matrix_index_list[i] = i;
                }
                mMatrixIndexTable[starting_point_id] = point_matrix_index_list;

                mXi = new DenseMatrix(mDimension, 1, starting_point_value);
            }
            else
            {
                throw new ArgumentNullException();
            }
        }

        /// <summary>
        /// Assumption: point2 = point1 + transition_cost
        /// </summary>
        /// <param name="point1_id"></param>
        /// <param name="point2_id"></param>
        /// <param name="point1_to_point2_transition_cost"></param>
        public void AddTransitionInfo(int point1_id, int point2_id, double[] point1_to_point2_transition_cost)
        {
            int[] point1_index_list = GetMatrixIndexList(point1_id);
            int[] point2_index_list = GetMatrixIndexList(point2_id);

            //point1 = - point2 - transition_cost
            for (int i = 0; i < mDimension; ++i)
            {
                mOmega[point1_index_list[i], point1_index_list[i]] += 1;
                mOmega[point1_index_list[i], point2_index_list[i]] += (-1);
                mXi[point1_index_list[i], 0] += (-point1_to_point2_transition_cost[i]);
            }

            //point2 = + point2 + transition_cost
            for (int i = 0; i < mDimension; ++i)
            {
                mOmega[point2_index_list[i], point1_index_list[i]] += (-1);
                mOmega[point2_index_list[i], point2_index_list[i]] += 1;
                mXi[point1_index_list[i], 0] += point1_to_point2_transition_cost[i];
            }
        }

        /// <summary>
        /// Assumption: landmark = point + distance
        /// </summary>
        /// <param name="point_id"></param>
        /// <param name="landmark_id"></param>
        /// <param name="distance_from_point_to_landmark"></param>
        public void AddLandmarkDistanceInfo(int point_id, int landmark_id, double[] distance_from_point_to_landmark)
        {
            AddTransitionInfo(point_id, landmark_id, distance_from_point_to_landmark);
        }

        public int[] GetMatrixIndexList(int point_id)
        {
            if (mMatrixIndexTable.ContainsKey(point_id))
            {
                return mMatrixIndexTable[point_id];
            }
            else
            {
                int row_count = mOmega.RowCount;
                int[] point_index_list = new int[mDimension];
                for (int i = 0; i < mDimension; ++i)
                {
                    point_index_list[i] = i + row_count;
                }
                mMatrixIndexTable[point_id] = point_index_list;
                ExpandMatrices();
                return point_index_list;
            }
        }

        public void ExpandMatrices()
        {
            int row_count = mOmega.RowCount;
            double[,] newOmega = new double[row_count + 1, row_count + 1];
            double[] newXi = new double[row_count + 1];
            for (int i = 0; i < row_count; ++i)
            {
                for (int j = 0; j < row_count; ++j)
                {
                    newOmega[i, j] = mOmega[i, j];
                }
                newXi[i] = mXi[i, 0];
            }
            mOmega = new DenseMatrix(newOmega);
            mXi = new DenseMatrix(row_count+1, 1, newXi);
        }

        /// <summary>
        /// return the estimated locations of each point and each landmark
        /// </summary>
        /// <returns>the key is the point or landmark id; the value is the location of the point or landmark</returns>
        public double[] Estimate()
        {
            int state_count = mXi.RowCount;
            Matrix<double> estimated_states = mOmega.Inverse().Multiply(mXi);
            double[] states = new double[state_count];
            for (int i = 0; i < state_count; ++i)
            {
                states[i] = estimated_states[i, 0];
            }
            Dictionary<int, double[]> result = new Dictionary<int, double[]>();
            foreach (int id in mMatrixIndexTable.Keys)
            {
                int[] matrix_index_list = mMatrixIndexTable[id];
                result[id] = new double[mDimension];
                for (int i = 0; i < mDimension; ++i)
                {
                    result[id][i] = states[matrix_index_list[i]];
                }
            }
            return states;
        }
    }
}
