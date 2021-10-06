## Case study 1

Studies the compression and segmentation of a long-running trajectory (timestamped geospatial data) into smaller separate and meaningful trajectories that go between origins and destinations identified by geofenced regions.

### Data: `data/case_study_1.json`

The file contains a Pandas dataframe saved as a `.json` file with orientation 'split' (data should be read using the function call `pd.read_json(filepath, orient="split")`). The dataframe holds a single synthetically generated normal trajectory going back and forth between an origin and a destination. The trajectory is generated to simulate a person walking between two locations and staying at each location for an extended period of time (15 min).

## Case study 2

Studies the ability of a proposed anomaly detection method to detect anomalous geospatial trajectories based on historical trajectory data. 

### Data: `data/case_study_2.json`

The file contains a Pandas dataframe saved as a `.json` file with orientation 'split' (data should be read using the function call `pd.read_json(filepath, orient="split")`). The dataframe holds a total of 222 synthetically generated normal and anomalous trajectories going between 7 different origins and destinations resulting in a variety of different trajectories. The trajectories generated as being normal are 199 while those being anomalous are 23.

## Case study 3

TODO