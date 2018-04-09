/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lenskit.mooc.svd;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.*;
import org.codehaus.groovy.runtime.powerassert.SourceText;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.entities.CommonAttributes;
import org.lenskit.data.ratings.Rating;
import org.lenskit.mooc.svd.SVDModelBuilder.MyRating;
import org.lenskit.util.keys.KeyIndex;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class MySingularValueDecomposition {

    private final int m;

    private final int n;


    private HashMap<Long, Double> itemPopularity;
    private int featureCount ;
    RealMatrix userMatrix;
    RealMatrix itemMatrix;
    RealMatrix originalMatrix;
    double[][] ratingData ;
    List<MyRating> ratingList;
    KeyIndex itemIndex;
    double TARGET_POP = 2.5;
    double thresholdPopularity;
    public MySingularValueDecomposition(RealMatrix matrix, int featureCount, HashMap<Long, Double> itemPop, int noOfUsers, int noOfItems, KeyIndex itemIndex, Double thresholdPopularity) {
        this.itemIndex = itemIndex;
        this.ratingList = ratingList;
        this.itemPopularity = itemPop;
        this.featureCount = featureCount;
        this.n = matrix.getColumnDimension();
        this.m = matrix.getRowDimension();
        this.thresholdPopularity = thresholdPopularity;
        userMatrix = MatrixUtils.createRealMatrix(noOfUsers, featureCount);
        Random rand = new Random();
        for(int i = 0; i< noOfUsers; i++) {
            for(int j = 0; j<featureCount; j++) {
                userMatrix.setEntry(i,j, rand.nextDouble()/10.0);
            }
        }
        itemMatrix = MatrixUtils.createRealMatrix(noOfItems, featureCount);
        for(int i = 0; i< noOfItems; i++) {
            for(int j = 0; j<featureCount; j++) {
                itemMatrix.setEntry(i,j, rand.nextDouble()/10.0);
            }
        }
        originalMatrix = matrix;
        ratingData = matrix.getData();
        computeSVD();
    }

    public void computeSVD(){

        int MAX_ITERATION =45;
        double alpha = 0.002;
        double beta = 0.02;
        double error = 0.0;
        double totalerror = 0.0;
        double weight = 0.5;
        double totalObjective = 0.0;
        int count = 0;
        for(int step = 1; step <= MAX_ITERATION; step++){
            totalerror = 0.0;
            count = 0;
            totalObjective = 0.0;

            for(int i=0; i < ratingData.length; i++){
                for(int j=0; j<ratingData[0].length; j++){
                    double rating = ratingData[i][j];
                    double objective = 0.0;
                    if(rating != 0){
                        count++;
                        double[] u = userMatrix.getRow(i);
                        double[] v = itemMatrix.getRow(j);
//                        error = Math.pow((rating - dotProduct(u,v)),2); //
                        long itemId = itemIndex.getKey(j);
                        double itemPop = itemPopularity.get(itemId);
//                        if(itemPop >= thresholdPopularity)
//                            rating = rating - 1;
                        error = rating - dotProduct(u,v);
                        totalerror += Math.pow(error,2);
                        //Math.pow(dotProduct(u, v), 2);

                        //weight = weight + alpha * (Math.pow(itemPopularity.get(itemId)-TARGET_POP, 2) - Math.pow(error , 2));
                        if(dotProduct(u,v) > 5.0 || itemPop > 5) {
                            for (int k = 0; k < featureCount; k++) {
//                            u[k] = u[k] + alpha * (2 * weight * error * v[k] - beta * u[k]);
//                            v[k] = v[k] + alpha * (2 * weight * error * u[k] - beta * v[k]);
                                u[k] = u[k] + alpha * (2 * error * v[k] - v[k] * (itemPop - 4.0) - beta * u[k]);
                                v[k] = v[k] + alpha * (2 * error * u[k] - u[k] * (itemPop - 4.0) - beta * v[k]);
                            }
                        }
                        else {
                            for(int k=0; k<featureCount; k++){
                                u[k] = u[k] + alpha * (2 * error * v[k] - beta * u[k]);
                                v[k] = v[k] + alpha * (2 * error * u[k] - beta * v[k]);
                            }
                        }

                        userMatrix.setRow(i, u);
                        itemMatrix.setRow(j, v);

                        //CHANGES REQUIRED HERE AND OTHER PLACES ABOVE
//                        if(itemPop >= thresholdPopularity)
//                            objective = weight * Math.pow(rating-dotProduct(u, v), 2) + (1 - weight) * Math.pow(itemPop - TARGET_POP, 2);
//                        totalObjective += objective;
                    }

                }
                //System.out.format("\n i:%d  Objective :%f  weight : %f",i, totalerror, weight);


            }
//            RealMatrix predMatrix = userMatrix.multiply((itemMatrix).transpose());
//            computeError(predMatrix);
            System.out.format("\nIteration : %d over , Error : %f", step, Math.sqrt(totalerror / (count)));
            if(Math.abs(totalerror) <= 0.5)
                break;
        }

        System.out.format("\nTrained at Error : %f", totalerror);
    }

    private double dotProduct(double[] u, double[] v){
        double result = 0.0;

        for(int i=0; i < u.length; i++){
            result += (u[i] * v[i]);
        }

        return result;
    }

    public RealMatrix getUserMatrix(){
        return userMatrix;
    }

    public RealMatrix getItemMatrix(){
        return itemMatrix;
    }

    /**
     * Returns the matrix U of the decomposition.
     * <p>U is an orthogonal matrix, i.e. its transpose is also its inverse.</p>
     * @return the U matrix
     * @see #getUT()
     */

}
