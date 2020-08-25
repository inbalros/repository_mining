from enum import Enum, auto


class DataType(Enum):
    CheckstyleDataType = "checkstyle"
    SourceMonitorFilesDataType = "source_monitor_files"
    SourceMonitorDataType = "source_monitor"
    CKDataType = "ck"
    MoodDataType = "mood"
    HalsteadDataType = "halstead"
    BuggedDataType = "bugged"
    BuggedMethodsDataType = "bugged_methods"


class DataName:
    def __init__(self, name, data_type: DataType, column_name, description=""):
        self.name = name
        self.data_type = data_type
        self.data_type_value = data_type.value
        self.column_name = column_name
        self.description = description

    def as_data_dict(self):
        return {'data_value': self.name, 'data_type': self.data_type_value, 'data_column': self.column_name}

    def as_description_dict(self):
        return {"feature_name": self.name, "feature_group": self.data_type_value, "column_name": self.column_name, "description": self.description}


class DataNameEnum(Enum):
    # TODO create accessors for each enumeration
    Bugged = DataName("Bugged", DataType.BuggedDataType, "is_buggy")
    BuggedMethods = DataName("BuggedMethods", DataType.BuggedMethodsDataType, "is_method_buggy")

    NCSSForThisFile = DataName("NCSSForThisFile", DataType.CheckstyleDataType, "NCSS_for_this_file")
    NestedIfElseDepth = DataName("NestedIfElseDepth", DataType.CheckstyleDataType, "Nested_if-else_depth")
    BooleanExpressionComplexity = DataName("BooleanExpressionComplexity", DataType.CheckstyleDataType, "Boolean_expression_complexity")
    CyclomaticComplexity = DataName("CyclomaticComplexity", DataType.CheckstyleDataType, "Cyclomatic_Complexity")
    NCSSForThisMethod = DataName("NCSSForThisMethod", DataType.CheckstyleDataType, "NCSS_for_this_method")
    NPathComplexity = DataName("NPathComplexity", DataType.CheckstyleDataType, "NPath_Complexity")
    ThrowsCount = DataName("ThrowsCount", DataType.CheckstyleDataType, "Throws_count")
    NCSSForThisClass = DataName("NCSSForThisClass", DataType.CheckstyleDataType, "NCSS_for_this_class")
    NumberOfProtectedMethod = DataName("NumberOfProtectedMethod", DataType.CheckstyleDataType, "Number_of_protected_methods")
    NumberOfPackageMethod = DataName("NumberOfPackageMethod", DataType.CheckstyleDataType, "Number_of_package_methods")
    NumberOfPrivateMethod = DataName("NumberOfPrivateMethod", DataType.CheckstyleDataType, "Number_of_private_methods")
    ExecutableStatementCount = DataName("ExecutableStatementCount", DataType.CheckstyleDataType, "Executable_statement_count")
    MethodLength = DataName("MethodLength", DataType.CheckstyleDataType, "Method_length")
    FileLength = DataName("FileLength", DataType.CheckstyleDataType, "File_length")
    AnonymousInnerClassLength = DataName("AnonymousInnerClassLength", DataType.CheckstyleDataType, "Anonymous_inner_class_length")
    NumberOfMethods_Checkstyle = DataName("NumberOfMethods_Checkstyle", DataType.CheckstyleDataType, "Total_number_of_methods")
    NumberOfPublicMethods_Checkstyle = DataName("NumberOfPublicMethods_Checkstyle", DataType.CheckstyleDataType, "Number_of_public_methods")
    ClassFanOutComplexity = DataName("ClassFanOutComplexity", DataType.CheckstyleDataType, "Class_Fan-Out_Complexity")
    NestedTryDepth = DataName("NestedTryDepth", DataType.CheckstyleDataType, "Nested_try_depth")
    ClassDataAbstractionCoupling = DataName("ClassDataAbstractionCoupling", DataType.CheckstyleDataType, "Class_Data_Abstraction_Coupling")
    NestedForDepth = DataName("NestedForDepth", DataType.CheckstyleDataType, "Nested_for_depth")

    # TODO SourceMonitorColumns - Need windows to do this
    SourceMonitorComplexity = DataName("SourceMonitorComplexity", DataType.SourceMonitorDataType, "Complexity")
    SourceMonitorStatements = DataName("SourceMonitorStatements", DataType.SourceMonitorDataType, "Statements")
    SourceMonitorMaximumDepth = DataName("SourceMonitorMaximumDepth", DataType.SourceMonitorDataType, "Maximum Depth")
    SourceMonitorCalls = DataName("SourceMonitorCalls", DataType.SourceMonitorDataType, "Calls")

    SourceMonitorLines = DataName("SourceMonitorCalls", DataType.SourceMonitorFilesDataType, "Lines")
    SourceMonitorFileStatements = DataName("SourceMonitorFileStatements", DataType.SourceMonitorFilesDataType, "FileStatements")
    MethodCallStatements = DataName("MethodCallStatements", DataType.SourceMonitorFilesDataType, "Method Call Statements")
    PercentLinesWithComments = DataName("PercentLinesWithComments", DataType.SourceMonitorFilesDataType, "Percent Lines with Comments")
    ClassesandInterfaces = DataName("ClassesandInterfaces", DataType.SourceMonitorFilesDataType, "Classes and Interfaces")
    MethodsperClass = DataName("MethodsperClass", DataType.SourceMonitorFilesDataType, "Methods per Class")
    AverageStatementsperMethod = DataName("AverageStatementsperMethod", DataType.SourceMonitorFilesDataType, "Average Statements per Method")
    MaximumComplexity = DataName("MaximumComplexity", DataType.SourceMonitorFilesDataType, "Maximum Complexity*")
    MaximumBlockDepth = DataName("MaximumBlockDepth", DataType.SourceMonitorFilesDataType, "Maximum Block Depth")
    AverageBlockDepth = DataName("AverageBlockDepth", DataType.SourceMonitorFilesDataType, "Average Block Depth")
    AverageComplexity = DataName("AverageComplexity", DataType.SourceMonitorFilesDataType, "Average Complexity*")
    Statementsatblocklevel0 = DataName("Statementsatblocklevel0", DataType.SourceMonitorFilesDataType, "Statements at block level 0")
    Statementsatblocklevel1 = DataName("Statementsatblocklevel1", DataType.SourceMonitorFilesDataType, "Statements at block level 1")
    Statementsatblocklevel2 = DataName("Statementsatblocklevel2", DataType.SourceMonitorFilesDataType, "Statements at block level 2")
    Statementsatblocklevel3 = DataName("Statementsatblocklevel3", DataType.SourceMonitorFilesDataType, "Statements at block level 3")
    Statementsatblocklevel4 = DataName("Statementsatblocklevel4", DataType.SourceMonitorFilesDataType, "Statements at block level 4")
    Statementsatblocklevel5 = DataName("Statementsatblocklevel5", DataType.SourceMonitorFilesDataType, "Statements at block level 5")
    Statementsatblocklevel6 = DataName("Statementsatblocklevel6", DataType.SourceMonitorFilesDataType, "Statements at block level 6")
    Statementsatblocklevel7 = DataName("Statementsatblocklevel7", DataType.SourceMonitorFilesDataType, "Statements at block level 7")
    Statementsatblocklevel8 = DataName("Statementsatblocklevel8", DataType.SourceMonitorFilesDataType, "Statements at block level 8")
    Statementsatblocklevel9 = DataName("Statementsatblocklevel9", DataType.SourceMonitorFilesDataType, "Statements at block level 9")

    IsConstructor = DataName("IsConstructor", DataType.CKDataType, "constructor")
    CBO = DataName("CBO", DataType.CKDataType, "cbo")
    WMC_CK = DataName("WMC_CK", DataType.CKDataType, "wmc")
    RFC = DataName("RFC", DataType.CKDataType, "rfc")
    LOCMethod_CK = DataName("LOCMethod_CK", DataType.CKDataType, "loc")
    Returns = DataName("Returns", DataType.CKDataType, "returns")
    NumberOfVariables = DataName("NumberOfVariables", DataType.CKDataType, "variables")
    NumberOfParameters_CK = DataName("NumberOfParameters_CK", DataType.CKDataType, "parameters")
    NumberOfLoops = DataName("NumberOfLoops", DataType.CKDataType, "loopQty")
    NumberOfComparisons = DataName("NumberOfComparisons", DataType.CKDataType, "comparisonsQty")
    NumberOfTryCatch = DataName("NumberOfTryCatch", DataType.CKDataType, "tryCatchQty")
    NumberOfParenthesizedExps = DataName("NumberOfParenthesizedExps", DataType.CKDataType, "parenthesizedExpsQty")
    NumberOfStringLiterals = DataName("NumberOfStringLiterals", DataType.CKDataType, "stringLiteralsQty")
    NumberOfNumbers = DataName("NumberOfNumbers", DataType.CKDataType, "numbersQty")
    NumberOfAssignments = DataName("NumberOfAssignments", DataType.CKDataType, "assignmentsQty")
    NumberOfMathOperations = DataName("NumberOfMathOperations", DataType.CKDataType, "mathOperationsQty")
    MaxNumberOfNestedBlocks = DataName("MaxNumberOfNestedBlocks", DataType.CKDataType, "maxNestedBlocks")
    NumberOfAnonymousClasses = DataName("NumberOfAnonymousClasses", DataType.CKDataType, "anonymousClassesQty")
    NumberOfInnerClasses = DataName("NumberOfInnerClasses", DataType.CKDataType, "innerClassesQty")
    NumberOfLambdas = DataName("NumberOfLambdas", DataType.CKDataType, "lambdasQty")
    NumberOfUniqueWords = DataName("NumberOfUniqueWords", DataType.CKDataType, "uniqueWordsQty")
    NumberOfModifiers = DataName("NumberOfModifiers", DataType.CKDataType, "modifiers")
    NumberOfLogStatements = DataName("NumberOfLogStatements", DataType.CKDataType, "logStatementsQty")

    NumberOfAncestors = DataName("NumberOfAncestors", DataType.MoodDataType, "numberOfAncestors")
    NumberOfSubclasses = DataName("NumberOfSubclasses", DataType.MoodDataType, "numberOfSubclasses")
    NumberOfPrivateAttributes = DataName("NumberOfPrivateAttributes", DataType.MoodDataType, "numbeOfPrivateAttributes")
    NumberOfProtectedAttributes = DataName("NumberOfProtectedAttributes", DataType.MoodDataType, "numberOfProtectedAttributes")
    NumberOfPublicAttributes = DataName("NumberOfPublicAttributes", DataType.MoodDataType, "numberOfPublicAttributes")
    NumberOfAttributes = DataName("NumberOfAttributes", DataType.MoodDataType, "numberOfAttributes")
    NumberOfCoupledClasses = DataName("NumberOfCoupledClasses", DataType.MoodDataType, "numberOfCoupledClasses")
    Cohesion = DataName("Cohesion", DataType.MoodDataType, "cohesion")
    NumberOfMethods_Mood = DataName("NumberOfMethods_Mood", DataType.MoodDataType, "numberOfMethods")
    NumberPublicMethods = DataName("NumberPublicMethods", DataType.MoodDataType, "numberPublicMethods")
    NumberUserDefinedAttributes = DataName("NumberUserDefinedAttributes", DataType.MoodDataType, "numberUserDefinedAttributes")
    NumberOfInheritedMethods = DataName("NumberOfInheritedMethods", DataType.MoodDataType, "numberOfInheritedMethods")
    NumberOfPolymorphicMethods = DataName("NumberOfPolymorphicMethods", DataType.MoodDataType, "numberOfPolymorphicMethods")

    TotalNumberOfOperators = DataName("TotalNumberOfOperators", DataType.HalsteadDataType, "getTotalOperatorsCnt")
    NumberOfDistinctOperators = DataName("NumberOfDistinctOperators", DataType.HalsteadDataType, "getDistinctOperatorsCnt")
    TotalNumberOfOperands = DataName("TotalNumberOfOperands", DataType.HalsteadDataType, "getTotalOparandsCnt")
    NumberOfDistinctOperands = DataName("NumberOfDistinctOperands", DataType.HalsteadDataType, "getDistinctOperandsCnt")
    Length = DataName("Length", DataType.HalsteadDataType, "getLength")
    Vocabulary = DataName("Vocabulary", DataType.HalsteadDataType, "getVocabulary")
    Volume = DataName("Volume", DataType.HalsteadDataType, "getVolume")
    Difficulty = DataName("Difficulty", DataType.HalsteadDataType, "getDifficulty")
    Effort = DataName("Effort", DataType.HalsteadDataType, "getEffort")
