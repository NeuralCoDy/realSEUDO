#include <stdio.h>

#include "mex.hpp"
#include "mexAdapter.hpp"
#include "seudo.hpp"
#include "strprintf.hpp"

using namespace matlab::data;
using matlab::mex::ArgumentList;

// This is a common wrapper that unifies multiple SEUDO and FISTA-related
// tools.  It looks like each native command requires a separate mex file,
// which means that they have to either each include all the common files, or
// link with a common shared library. Creating a shared library is different on
// Mac and Linux, so for portability it's easier to avoid that. And if state is
// to be saved in the global/static variables across the calls, this state
// cannot be in different shared libraries (and mex files are shared
// libraries).  So the solution taken here is to make one command wrapper
// function, and then differentiate based on the first string argument of the
// function into mutiple different functions.


class MexFunction : public matlab::mex::Function {
public:
	// Inputs:
	//   string command - sub-command to call, which handles the rest of args
	// Outputs enbedded into the wrapper:
	//   string log - whatever help messages
	enum InArgIdx {
		InCommand,
	};
	enum OutHelpArgIdx {
		OutLog,
	};

	void operator()(ArgumentList outputs, ArgumentList inputs) {
		std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
		ArrayFactory factory;

		std::string log;

		if (inputs.size() <= InCommand) {
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("the first argument must be a command name") }));
			return;
		}

		const auto &cmd = inputs[InCommand];
		auto cmdDims = cmd.getDimensions();
		log += strprintf("Command dimensions:");
		for (int i = 0; i < cmdDims.size(); i++) {
			log += strprintf(" %d", (int)cmdDims[i]);
		}
		log += strprintf("\n");

		if (cmd.getType() == ArrayType::CHAR) {
			log += strprintf("got a char array\n");
			std::string scmd = cmd[0][0];
			log += scmd;
		} else if (cmd.getType() == ArrayType::MATLAB_STRING) {
			log += strprintf("got a string\n");
			std::string scmd = cmd[0];
			log += scmd;
		} else {
			log += strprintf("got arg type %d\n", (int)cmd.getType());
		}

		if (outputs.size() > OutLog) {
			outputs[OutLog] = factory.createScalar(log.c_str());
		}

	}

};

