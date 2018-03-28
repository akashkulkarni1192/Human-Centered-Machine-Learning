package org.lenskit.mooc.svd;

import it.unimi.dsi.fastutil.longs.LongList;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.apache.commons.math3.linear.RealVector;
import org.codehaus.groovy.runtime.powerassert.SourceText;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.lenskit.LenskitRecommender;
import org.lenskit.api.Recommender;
import org.lenskit.eval.traintest.AlgorithmInstance;
import org.lenskit.eval.traintest.DataSet;
import org.lenskit.eval.traintest.TestUser;
import org.lenskit.eval.traintest.metrics.MetricColumn;
import org.lenskit.eval.traintest.metrics.MetricResult;
import org.lenskit.eval.traintest.metrics.TypedMetricResult;
import org.lenskit.eval.traintest.recommend.ListOnlyTopNMetric;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;
import java.io.IOException;
import java.util.HashMap;
import java.io.FileWriter;
import java.util.Map;
import java.util.Scanner;

public class ListPopularityMetric extends ListOnlyTopNMetric<ListPopularityMetric.Context> {

    //    private HashMap<Double, SVDModel> modelMap;
    private final String freqFileName = "something.csv";
    private static final String FILE_HEADER = "MovieId,Popularity,PopularityWeight,RecFrequency";
    private static final String NEW_LINE_SEPARATOR = "\n";
    private static final String COMMA_DELIMITER = ",";

    //    private SVDModel model = null;
    @Inject
    public ListPopularityMetric() {
        super(ListPopularityMetric.UserResult.class, ListPopularityMetric.AggregateResult.class);
    }

    @Nullable
    @Override
    public Context createContext(AlgorithmInstance algorithm, DataSet dataSet, Recommender recommender) {
        return new Context(dataSet.getAllItems(), (LenskitRecommender) recommender);
    }

    @Nonnull
    @Override
    synchronized public MetricResult getAggregateMeasurements(ListPopularityMetric.Context context) {
        return new ListPopularityMetric.AggregateResult(context);
    }

    @Nonnull
    @Override
    public MetricResult measureUser(TestUser user, int targetLength, LongList recommendations, Context context) {
        SVDModel model = context.recommender.get(SVDModel.class);
        Double listPopularity = 0.0;
        for(Long item : recommendations){
            listPopularity += Math.pow(model.getItemPopularity(item), 2);
        }
        ListPopularityMetric.UserResult result = new ListPopularityMetric.UserResult(listPopularity);
        context.addUser(result);
        return result;
    }

    public static class UserResult extends TypedMetricResult {
        @MetricColumn("ListPopularity")
        public final Double listPopularity;

        public UserResult(Double val) {
            listPopularity = val;
        }

        public Double getIlsValue() { return listPopularity; }

    }

    public static class AggregateResult extends TypedMetricResult {

        @MetricColumn("ListPopularity")
        public final Double listPopularity;

        public AggregateResult(ListPopularityMetric.Context accum) {
            this.listPopularity = accum.allMean.getMean();
        }
    }

    public static class Context {
        private final LongSet universe;
        private final MeanAccumulator allMean = new MeanAccumulator();
        private final LenskitRecommender recommender;

        Context(LongSet universe, LenskitRecommender recommender) {
            this.universe = universe;
            this.recommender = recommender;
        }

        void addUser(ListPopularityMetric.UserResult ur) {
            allMean.add(ur.getIlsValue());
        }
    }
}
