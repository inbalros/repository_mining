import os
from enum import Enum

from config import Config


class ProjectName(Enum):
    Camel = "camel", "CAMEL"
    Hadoop = "hadoop", "HADOOP"
    Flink = "flink", "FLINK"
    Kafka = "kafka", "KAFKA"
    OpenMeetings = "openmeetings", "OPENMEETINGS"
    Karaf = "karaf", "KARAF"
    Hbase = "hbase", "HBASE"
    Beam = "beam", "BEAM"
    Submarine = "submarine", "SUBMARINE"
    CloudStack = "cloudstack", "CLOUDSTACK"
    Netbeans = "netbeans", "NETBEANS"
    UimaRuta = "uima-ruta", "UIMA"
    Lucene = "lucene-solr", "LUCENE"
    DeltaSpike = "deltaspike", "DELTASPIKE"
    JackrabbitOak = "jackrabbit-oak", "OAK"
    Pulsar = "pulsar", "PULSAR"
    Ofbiz = "ofbiz", "OFBIZ"
    Cayenne = "cayenne", "CAY"
    CommonsCodec = "commons-codec", "CODEC"
    Parquet = "parquet-mr", "PARQUET"
    Kylin = "kylin", "KYLIN"
    RocketMQ = "rocketmq", "ROCKETMQ"
    Ignite = "ignite", "IGNITE"
    Dubbo = "dubbo", "DUBBO"
    Hive = "hive", "HIVE"
    CommonsValidator = "commons-validator", "VALIDATOR"
    Groovy = "groovy", "GROOVY"
    Surefire = "maven-surefire", "SUREFIRE"
    Syncope = "syncope", "SYNCOPE"
    CommonsMath = "commons-math", "MATH"
    CommonsImaging = "commons-imaging", "IMAGING"
    Tomcat = "tomcat", "MTOMCAT"
    Plc4x = "plc4x", "PLC4X"
    Atlas = "atlas", "ATLAS"
    Struts = "struts", "STR"
    Tika = "tika", "TIKA"
    ServiceComb = "servicecomb-java-chassis", "SCB"
    Ranger = "ranger", "RANGER"
    Cassandra = "cassandra", "CASSANDRA"
    Synapse = "synapse", "SYNAPSE"
    CXF = "cxf", "CXF"
    Metron = "metron", "METRON"
    Avro = "avro", "AVRO"
    Nifi = "nifi", "NIFI"
    Bookkeeper = "bookkeeper", "BOOKKEEPER"
    Clerezza = "clerezza", "CLEREZZA"
    SystemML = "systemml", "SYSTEMML"
    AsterixDB = "asterixdb", "ASTERIXDB"
    Unomi = "unomi", "UNOMI"
    Maven = "maven", "MNG"
    Fineract = "fineract", "FINERACT"
    Zeppelin = "zeppelin", "ZEPPELIN"
    CommonsCollections = "commons-collections", "COLLECTIONS"
    Jena = "jena", "JENA"
    Calcite = "calcite", "CALCITE"
    ActiveMQArtemis = "activemq-artemis", "ARTEMIS"
    Tez = "tez", "TEZ"
    CommonsLang = "commons-lang", "LANG"
    ActiveMQ = "activemq", "AMQ"
    Curator = "curator", "CURATOR"
    Phoenix = "phoenix", "PHOENIX"
    Samza = "samza", "SAMZA"
    Nutch = "nutch", "NUTCH"
    QpidJMS = "qpid-jms", "QPIDJMS"
    DirectoryKerby = "directory-kerby", "DIRKRB"
    Juneau = "juneau", "JUNEAU"
    Bigtop = "bigtop", "BIGTOP"
    MyFacesTobago = "myfaces-tobago", "TOBAGO"
    Isis = "isis", "ISIS"
    Wicket = "wicket", "WICKET"
    Santuario = "santuario-java", "SANTUARIO"
    Helix = "helix", "HELIX"
    Storm = "storm", "STORM"
    Airavata = "airavata", "AIRAVATA"
    MyFaces = "myfaces", "MYFACES"
    CommonsDBCP = "commons-dbcp", "DBCP"
    CommonsVFS = "commons-vfs", "VFS"
    OpenNLP = "opennlp", "OPENNLP"
    Tomee = "tomee", "TOMEE"
    TinkerPop = "tinkerpop", "TINKERPOP"
    DirectoryServer = "directory-server", "DIRSERVER"
    CommonsCompress = "commons-compress", "COMPRESS"
    Accumulo = "accumulo", "ACCUMULO"
    Giraph = "giraph", "GIRAPH"
    Johnzon = "johnzon", "JOHNZON"
    JClouds = "jclouds", "JCLOUDS"
    Griffin = "griffin", "GRIFFIN"
    ManifoldCF = "manifoldcf", "CONNECTORS"
    Shiro = "shiro", "SHIRO"
    Knox = "knox", "KNOX"
    Drill = "drill", "DRILL"
    Crunch = "crunch", "CRUNCH"
    CommonsIO = "commons-io", "IO"
    CommonsCLI = "commons-cli", "CLI"
    Jackrabbit = "jackrabbit", "JCR"
    OpenWebBeans = "openwebbeans", "OWB"
    FOP = "xmlgraphics-fop", "FOP"
    Tajo = "tajo", "TAJO"
    CommonsEmail = "commons-email", "EMAIL"
    DirectoryStudio = "directory-studio", "DIRSTUDIO"
    Tapestry5 = "tapestry-5", "TAP5"
    Archiva = "archiva", "MRM"

    def github(self):
        return self.value[0]

    def jira(self):
        return self.value[1]

    def path(self):
        return os.path.join(Config().config['REPO']['RepoDir'], self.value[0])