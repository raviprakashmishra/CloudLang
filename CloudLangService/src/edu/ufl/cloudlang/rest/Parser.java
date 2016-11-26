/**
 * This class provides the endpoint for the Parsing task.
 */
package edu.ufl.cloudlang.rest;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.Response;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * @author Sayak Biswas(54584911)
 *
 */

@Path("/parse")
public class Parser {
	@GET
	@Produces("application/json")
	public Response parseText() throws JSONException {
		System.out.println("In parse");
		JSONObject json = new JSONObject();
		json.put("testparsekey", "testparsevalue");
		
		String result = "@Produces(\"application/json\") Output: \n\n Parse result: \n\n" + json;
		return Response.status(200).entity(result).build();
	}
	
	@Path("{text}")
	@GET
	@Produces("application/json")
	public Response parseTextWithInput(@PathParam("text") String text) throws JSONException {
		System.out.println("Input Text :: " + text);
		
		FileOutputStream fileOutputStream = null;
		File file = null;
		try {
			file = new File("input.txt");
			fileOutputStream = new FileOutputStream(file);
			if(!file.exists()) {
				file.createNewFile();
			}
			byte[] inputInBytes = text.getBytes();
			fileOutputStream.write(inputInBytes);
			fileOutputStream.flush();
		} catch (IOException ie) {
			ie.printStackTrace();
		} finally {
			if(fileOutputStream != null) {
				try {
					fileOutputStream.close();
				} catch (IOException ie2) {
					ie2.printStackTrace();
				}
			}
		}
		
		String stanfordParserPath = "/home/sayak/Workspace/nndep-torch7/stanford-parser-full-2013-11-12/stanford-parser.jar";
		String lexicalParserClassName = "edu.stanford.nlp.parser.lexparser.LexicalizedParser";
		String grammaticalStructureClassName = "edu.stanford.nlp.trees.EnglishGrammaticalStructure";
		String outputFormat = "penn";
		String lexicalModel = "/home/sayak/Workspace/nndep-torch7/englishPCFG.ser.gz";
		String outputPath = "/home/sayak/Workspace/nndep-torch7/output.mrg";
		String outputDependenciesPath = "/home/sayak/Workspace/nndep-torch7/output.mrg.dep";
		String dependencyParserPath = "/home/sayak/Workspace/nndep-torch7/dep/parse.lua";
		String dependencyModelPath = "/home/sayak/Workspace/nndep-torch7/model.th7";
		String finalParseTreePath = "/home/sayak/Workspace/nndep-torch7/output.conll";
		String command = "java -mx200m -cp " + stanfordParserPath + " " 
							+ lexicalParserClassName 
							+ " -retainTMPSubcategories -outputFormat " + outputFormat + " " 
							+ lexicalModel + " " + "input.txt";
		Process process = null;
		String log = null;
		BufferedReader stdInput = null;
		BufferedWriter stdOutput = null;
		try {
			System.out.println("Will execute command " + command);
			process = Runtime.getRuntime().exec(command);
			if(process != null) {
				stdInput = new BufferedReader(new InputStreamReader(process.getInputStream()));
				stdOutput = new BufferedWriter(new FileWriter(outputPath));
				System.out.println("Process logs :: ");
				while ((log = stdInput.readLine()) != null) {
					System.out.println(log);
					stdOutput.write(log);
					stdOutput.newLine();
				}
				stdOutput.flush();
				stdInput.close();
				stdOutput.close();
			}
		} catch (IOException ie) {
			ie.printStackTrace();
		} finally {
			if(stdInput != null) {
				try {
					stdInput.close();
				} catch (IOException ie) {
					ie.printStackTrace();
				}
			}
			
			if(stdOutput != null) {
				try {
					stdOutput.close();
				} catch (IOException ie) {
					ie.printStackTrace();
				}
			}
		}
		BufferedReader stdInput2 = null;
		BufferedWriter stdOutput2 = null;
		Process process2 = null;
		String log2 = null;
		try {
			command = "java -cp " + stanfordParserPath + " " + grammaticalStructureClassName 
						+ " -maxLength 100 -basic -conllx -treeFile " +  outputPath;
			System.out.println("Will execute command " + command);
			process2 = Runtime.getRuntime().exec(command);
			stdInput2 = new BufferedReader(new InputStreamReader(process2.getInputStream()));
			stdOutput2 = new BufferedWriter(new FileWriter(outputDependenciesPath));
			System.out.println("Process logs :: ");
			while ((log2 = stdInput2.readLine()) != null) {
				System.out.println(log2);
				stdOutput2.write(log2);
				stdOutput2.newLine();
			}
			stdOutput2.flush();
			stdInput2.close();
			stdOutput2.close();
		} catch (IOException ie) {
			ie.printStackTrace();
		} finally {
			if(stdInput2 != null) {
				try {
					stdInput2.close();
				} catch (IOException ie) {
					ie.printStackTrace();
				}
			}
			
			if(stdOutput2 != null) {
				try {
					stdOutput2.close();
				} catch (IOException ie) {
					ie.printStackTrace();
				}
			}
		}
		
		try {
			command = "optirun th " + dependencyParserPath + " --rootLabel 'ROOT' --modelPath " + dependencyModelPath 
						+ " --input " + outputDependenciesPath + " --output " + finalParseTreePath + " --cuda";
			
			System.out.println("Will execute command " + command);
			Runtime.getRuntime().exec("/home/sayak/torch/install/bin/torch-activate");
			Runtime.getRuntime().exec("/bin/sh -c " + "\"" + command + "\"");
		} catch (IOException ie) {
			ie.printStackTrace();
		}
		System.out.println("Executed model");
		JSONObject json =  new JSONObject();
		json.put("inputText", text);
		StringBuilder parsedJSON = null;
		BufferedReader inputReader = null;
		String line = null;
		try {
			inputReader = new BufferedReader(new FileReader(new File(finalParseTreePath)));
			parsedJSON = new StringBuilder();
			while((line = inputReader.readLine()) != null) {
				parsedJSON.append(line);
				parsedJSON.append("\n");
			}
		} catch (FileNotFoundException re) {
			re.printStackTrace();
		} catch (IOException ie) {
			ie.printStackTrace();
		} finally {
			try {
				if(inputReader != null) {
					inputReader.close();
				}
			} catch (IOException ie) {
				ie.printStackTrace();
			}
		}
		json.put("parsedResult", parsedJSON);
		
		String result = "@Produces(\"application/json\") Output: \n\n Parse result: \n\n" + json;
		return Response.status(200).entity(result).build();
	}
}
